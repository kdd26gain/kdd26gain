import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. Independent Preprocessing & Dynamic Graph Generation (Entropy-based)
# -----------------------------------------------------------------------------

def fetch_price_data(region, start_date="2013-01-01", end_date="2025-03-31"):
    data_dir = "../data"
    csv_file = os.path.join(data_dir, f"stocks_{region}.csv")

    if not os.path.exists(csv_file):
        if os.path.exists(f"data/stocks_{region}.csv"):
            csv_file = f"data/stocks_{region}.csv"
        else:
            raise FileNotFoundError(f"Data file not found: {csv_file}")

    print(f"[Preprocess] Loading raw data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = pd.to_datetime(df['date'])

    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    if region == "KOR":
        df = df[df['symbol'] != '010130']
    if region == "CHN":
        df = df[df['symbol'] != 'SZSE: 002714']

    return df.sort_values(['symbol', 'date'])


def calculate_entropy_energy(windows, bins=10):
    """
    Calculate Signal Energy and Information Entropy for each window.
    windows: (Num_Windows, N, Lags)
    Output: Energy (Num_Windows, N), Entropy (Num_Windows, N)
    """
    # 1. Signal Energy: Sum(x^2)
    energy = (windows ** 2).sum(dim=2)  # (B, N)

    # 2. Entropy: -Sum(p log p)
    # Estimate probability p using histogram
    B, N, L = windows.shape
    entropy = torch.zeros(B, N, device=windows.device)

    min_val = windows.min(dim=2, keepdim=True)[0]
    max_val = windows.max(dim=2, keepdim=True)[0]
    # Avoid division by zero
    norm_windows = (windows - min_val) / (max_val - min_val + 1e-9)

    # Quantize to bins
    quantized = (norm_windows * bins).long().clamp(0, bins - 1)  # (B, N, L)

    # Count frequencies
    probs = torch.zeros(B, N, bins, device=windows.device)
    for i in range(L):
        val = quantized[:, :, i]  # (B, N)
        # One-hot encoding approach to count
        one_hot = F.one_hot(val, num_classes=bins).float()  # (B, N, bins)
        probs += one_hot

    probs = probs / L  # Normalize to probability

    # Entropy calculation (handle 0 log 0)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=2)  # (B, N)

    return energy, entropy


def build_mgdpr_dataset(region, lags=20, cache_dir="cache_mgdpr"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"mgdpr_data_{region}_l{lags}.pkl")
    if os.path.exists(cache_path):
        print(f"[Preprocess] Loading cached MGDPR dataset: {cache_path}")
        with open(cache_path, "rb") as f: return pickle.load(f)

    print(f"[Preprocess] Building new MGDPR dataset for {region}...")
    df = fetch_price_data(region)
    closes = df.pivot(index='date', columns='symbol', values='close').ffill()
    caps = df.pivot(index='date', columns='symbol', values='market_cap').ffill().fillna(0.0)

    returns = closes.pct_change().fillna(0.0)
    log_returns = np.log(closes / closes.shift(1)).fillna(0.0)

    print("[Preprocess] Generating Entropy-based Dynamic Graphs (MGDPR)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_gpu = torch.tensor(log_returns.values, device=device, dtype=torch.float32)

    # Create Windows: (Num_Windows, N, Lags)
    # Align: Graph at 't' uses data [t-lags : t]
    windows = X_gpu.unfold(0, lags, 1)  # (T-L+1, N, L)

    # Calculate Energy and Entropy
    E, H = calculate_entropy_energy(windows)  # (T-L+1, N)

    # Generate Adjacency Matrix
    # A_ij = (E_i / E_j) * exp(H_i - H_j)
    # Log space: log(A_ij) = log(E_i) - log(E_j) + (H_i - H_j)
    # A_ij = exp( log(E_i) - log(E_j) + H_i - H_j )

    log_E = torch.log(E + 1e-9)

    # Broadcasting to (B, N, N)
    # i is source (row), j is target (col). "Directed edge from vi to vj"
    # term1: log(E_i) - log(E_j) -> (B, N, 1) - (B, 1, N)
    term_E = log_E.unsqueeze(2) - log_E.unsqueeze(1)

    # term2: H_i - H_j
    term_H = H.unsqueeze(2) - H.unsqueeze(1)

    exponent = term_E + term_H
    adj_matrix = torch.exp(exponent)  # (B, N, N)

    # Pruning / Thresholding
    adj_matrix = torch.clamp(adj_matrix, 0, 10.0)  # Prevent explosion

    # Move to CPU
    adj_all = adj_matrix.cpu()

    data = {
        "X": torch.tensor(log_returns.values, dtype=torch.float32).unsqueeze(-1),
        "Y": torch.tensor(returns.values, dtype=torch.float32),
        "Caps": torch.tensor(caps.values, dtype=torch.float32),
        "Adj": adj_all,  # (T-L+1, N, N)
        "dates": closes.index,
        "symbols": closes.columns.tolist()
    }
    with open(cache_path, "wb") as f: pickle.dump(data, f)
    print(f"[Preprocess] Saved to {cache_path}")
    return data


# -----------------------------------------------------------------------------
# 2. MGDPR Model Architecture
# -----------------------------------------------------------------------------

class MultiRelationalGraphDiffusion(nn.Module):
    """
    Multi-relational Graph Diffusion
    H_l = Sigma( Conv( Concat( S * H * W ) ) )
    """

    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.K = K
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.conv1x1 = nn.Conv2d(K, 1, kernel_size=1)  # Aggregates K diffusion steps
        self.act = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # h: (B, N, D)
        # adj: (B, N, N)

        # 1. Transform features
        h_trans = self.W(h)  # (B, N, D_out)

        # 2. Diffusion Steps
        # S^k * H
        diffusions = []
        curr_h = h_trans

        # Step 0
        diffusions.append(curr_h)

        # Step 1 to K-1
        # Normalize Adj for stability (Random Walk normalization)
        # D^-1 A
        row_sum = adj.sum(dim=2, keepdim=True) + 1e-9
        adj_norm = adj / row_sum

        for k in range(1, self.K):
            curr_h = torch.bmm(adj_norm, curr_h)  # (B, N, D)
            diffusions.append(curr_h)

        # Stack: (B, K, N, D)
        stack = torch.stack(diffusions, dim=1)

        # Conv2d 1x1 to aggregate K steps -> (B, 1, N, D)
        # Treat (N, D) as spatial dimensions.
        out = self.conv1x1(stack)  # (B, 1, N, D)
        out = out.squeeze(1)  # (B, N, D)

        return self.act(out)


class ParallelRetention(nn.Module):
    """
    Parallel Retention Mechanism
    eta(Z) = GroupNorm( (Q K^T dot D) V )
    """

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.grp_norm = nn.GroupNorm(n_heads, d_model)  # GroupNorm over channels

        self.gamma = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        # x: (B*N, T, D) - Processing sequences
        Batch, T, D = x.shape

        Q = self.W_Q(x).view(Batch, T, self.n_heads, self.head_dim)
        K = self.W_K(x).view(Batch, T, self.n_heads, self.head_dim)
        V = self.W_V(x).view(Batch, T, self.n_heads, self.head_dim)

        # (B, H, T, D_h)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Q K^T -> (B, H, T, T)
        attn = torch.matmul(Q, K.transpose(-1, -2))

        # Decay Mask D_ij
        # D_ij = gamma^(i-j) if i>=j else 0
        indices = torch.arange(T, device=x.device)
        # (T, 1) - (1, T) -> (T, T) matrix of i-j
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)

        # Causal Mask & Decay
        mask = torch.tril(torch.ones(T, T, device=x.device))
        decay = (self.gamma ** diff) * mask

        # Apply Mask
        attn = attn * decay.unsqueeze(0).unsqueeze(0)  # Broadcast to (B, H, T, T)

        # Multiply V
        out = torch.matmul(attn, V)  # (B, H, T, D_h)

        # Reshape & GroupNorm
        out = out.permute(0, 2, 1, 3).reshape(Batch, T, D)
        # GroupNorm expects (N, C, L) usually
        out = out.transpose(1, 2)  # (B, D, T)
        out = self.grp_norm(out)
        out = out.transpose(1, 2)  # (B, T, D)

        return out


class MGDPR(nn.Module):
    """
    MGDPR Architecture
    """

    def __init__(self, d_feat=1, d_model=64, n_heads=4, dropout=0.1, layers=2):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model))  # Simple learnable PE

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'diffusion': MultiRelationalGraphDiffusion(d_model, d_model, K=3),
                'retention': ParallelRetention(d_model, n_heads),
                'norm': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model * 2, d_model),  # Concatenation reduction
                    nn.LeakyReLU(0.2)
                )
            }))

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, adj):
        # x: (B, N, T, F)
        # adj: (B, N, N)
        B, N, T, F = x.shape

        # 1. Flatten for Sequence Processing
        x_flat = x.reshape(B * N, T, F)
        h = self.input_proj(x_flat) + self.pos_enc[:, :T, :]

        # MGDPR Layers
        for layer in self.layers:
            # A. Parallel Retention (Intra-Stock)
            # h: (B*N, T, D)
            h_ret = layer['retention'](h)
            h_ret = layer['norm'](h_ret)

            # B. Graph Diffusion (Inter-Stock)
            # Shape: (B*N, T, D) -> Take last step T -> (B, N, D)
            h_last = h_ret[:, -1, :].reshape(B, N, -1)

            h_diff = layer['diffusion'](h_last, adj)  # (B, N, D)

            # C. Update & Skip Connection
            # Combine Retention output (Time) and Diffusion output (Graph)
            h_diff_exp = h_diff.unsqueeze(2).repeat(1, 1, T, 1).reshape(B * N, T, -1)

            combined = torch.cat([h_ret, h_diff_exp], dim=-1)
            h = layer['mlp'](combined)

        # Prediction
        # Take last step of final representation
        last_step = h[:, -1, :].reshape(B, N, -1)
        out = self.output_head(last_step)

        return out.squeeze(-1)


# -----------------------------------------------------------------------------
# 3. Experiment Runner & Metrics
# -----------------------------------------------------------------------------

def backtest_stats(returns_vec):
    if len(returns_vec) == 0: return 0.0, 0.0, 0.0
    r_arr = np.array(returns_vec)
    mean_ret = np.mean(r_arr)
    std_ret = np.std(r_arr) + 1e-9
    avol = std_ret * np.sqrt(252)
    asr = (mean_ret / std_ret) * np.sqrt(252)
    cum_ret = np.cumprod(1 + r_arr)
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / (peak + 1e-9)
    rmdd = np.abs(np.min(dd))
    return asr, rmdd, avol


def calc_cwmse(pred, true, caps, p_list=[3, 5, 10]):
    sorted_idx = np.argsort(caps)[::-1]
    metrics = {}
    for p in p_list:
        k = min(p, len(sorted_idx))
        if k == 0: metrics[p] = 0.0; continue
        top_k = sorted_idx[:k]
        w = caps[top_k]
        err = (pred[top_k] - true[top_k]) ** 2
        metrics[p] = np.sum(w * err) / (np.sum(w) + 1e-12)
    return metrics


def prepare_windows(data, lags):
    X, Y, Caps = data['X'], data['Y'], data['Caps']
    Adj = data['Adj']  # (T-L+1, N, N)
    T_total = X.shape[0]
    X_l, Y_l, C_l, A_l = [], [], [], []

    for i in range(lags, T_total):
        X_l.append(X[i - lags: i])
        Y_l.append(Y[i])
        C_l.append(Caps[i - 1])
        # Graph at time i-lags (start of window) or i (end)?
        # Preprocessing generated Adj aligned with windows starting at 0.
        # Window [i-lags:i] corresponds to Adj index (i-lags).
        A_l.append(Adj[i - lags])

    return torch.stack(X_l), torch.stack(Y_l), torch.stack(C_l), torch.stack(A_l), data['dates'][lags:]


def run_single_seed(args, seed, cached_data):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, Caps, Adj, dates = prepare_windows(cached_data, args.lags)

    train_from = pd.to_datetime(args.train_from)
    valid_from = pd.to_datetime(args.valid_from)
    test_from = pd.to_datetime(args.test_from)
    test_to = pd.to_datetime(args.test_to)

    train_mask = (dates >= train_from) & (dates < valid_from)
    val_mask = (dates >= valid_from) & (dates < test_from)
    test_mask = (dates >= test_from) & (dates <= test_to)

    if not train_mask.any(): return None

    def get_z(y):
        mu = y.mean(dim=1, keepdim=True)
        sig = y.std(dim=1, keepdim=True) + 1e-6
        return (y - mu) / sig

    Y_train_z = get_z(Y[train_mask])
    Y_val_z = get_z(Y[val_mask])

    train_ds = TensorDataset(X[train_mask], Y_train_z, Adj[train_mask])
    val_ds = TensorDataset(X[val_mask], Y_val_z, Adj[val_mask])
    test_ds = TensorDataset(X[test_mask], Y[test_mask], Caps[test_mask], Adj[test_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MGDPR(d_feat=X.shape[3], d_model=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None

    for _ in range(args.epochs):
        model.train()
        for bx, by, badj in train_loader:
            bx, by, badj = bx.to(device), by.to(device), badj.to(device)
            # bx: (B, T, N, F) -> permute to (B, N, T, F) for MGDPR
            bx = bx.permute(0, 2, 1, 3)
            optimizer.zero_grad()
            loss = criterion(model(bx, badj), by)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx, by, badj in val_loader:
                bx, by, badj = bx.to(device), by.to(device), badj.to(device)
                bx = bx.permute(0, 2, 1, 3)
                vloss += criterion(model(bx, badj), by).item()

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_weights = model.state_dict().copy()

    # Test
    model.load_state_dict(best_weights)
    model.eval()

    cwmse_sums = {3: 0, 5: 0, 10: 0}
    daily_rets = []
    mse_sum, mae_sum, count = 0, 0, 0
    SCALE = 100.0

    with torch.no_grad():
        for bx, by_raw, bcap, badj in test_loader:
            bx, badj = bx.to(device), badj.to(device)
            bx = bx.permute(0, 2, 1, 3)
            pred_z = model(bx, badj).cpu().numpy()
            true_raw = by_raw.numpy()
            caps_raw = bcap.numpy()

            for i in range(len(pred_z)):
                mu = true_raw[i].mean()
                sig = true_raw[i].std() + 1e-6
                pred_raw = pred_z[i] * sig + mu

                # Investment
                top5_idx = np.argsort(pred_z[i])[-5:]
                daily_rets.append(np.mean(true_raw[i][top5_idx]))

                # Metrics
                pred_pct = pred_raw * SCALE
                true_pct = true_raw[i] * SCALE
                mse_sum += np.mean((pred_pct - true_pct) ** 2)
                mae_sum += np.mean(np.abs(pred_pct - true_pct))

                c_res = calc_cwmse(pred_pct, true_pct, caps_raw[i])
                for k in cwmse_sums: cwmse_sums[k] += c_res[k]
                count += 1

    metrics = {f'CWMSE@{k}': v / count for k, v in cwmse_sums.items()}
    metrics['MSE'] = mse_sum / count
    metrics['MAE'] = mae_sum / count
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['ASR'], metrics['RMDD'], metrics['AVol'] = backtest_stats(daily_rets)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="S&P500")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--train_from", type=str, default="2013-01-01")
    parser.add_argument("--valid_from", type=str, default="2024-01-01")
    parser.add_argument("--test_from", type=str, default="2024-10-01")
    parser.add_argument("--test_to", type=str, default="2025-03-31")
    parser.add_argument("--lags", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    rmap = {"S&P500": "USA", "CSI300": "CHN", "EURO50": "EUR", "KP200": "KOR"}
    reg_code = rmap.get(args.region, args.region)

    data = build_mgdpr_dataset(reg_code, args.lags)

    results = []
    print(f"Running {args.n_seeds} seeds for {reg_code} (MGDPR)...")
    for s in range(args.n_seeds):
        res = run_single_seed(args, s, data)
        if res:
            results.append(res)
            print(f"Seed {s}: ASR={res['ASR']:.4f} | MSE={res['MSE']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (MGDPR)")
    print("-" * 80)
    df = pd.DataFrame(results)
    cols = ['ASR', 'RMDD', 'AVol', 'MSE', 'MAE', 'RMSE', 'CWMSE@3', 'CWMSE@5', 'CWMSE@10']
    for col in cols:
        if col in df.columns:
            mean_v = df[col].mean()
            std_v = df[col].std()
            print(f"{col:<10} | {mean_v:.4f} ± {std_v:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()