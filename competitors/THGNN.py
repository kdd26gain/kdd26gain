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
# 1. Independent Preprocessing for THGNN (Dynamic Graph Generation)
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


def build_thgnn_dataset(region, lags=20, cache_dir="cache_thgnn"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"thgnn_data_{region}_l{lags}.pkl")

    if os.path.exists(cache_path):
        print(f"[Preprocess] Loading cached THGNN dataset: {cache_path}")
        with open(cache_path, "rb") as f: return pickle.load(f)

    print(f"[Preprocess] Building new THGNN dataset for {region}...")
    df = fetch_price_data(region)

    # 1. Pivot Tables (T x N)
    closes = df.pivot(index='date', columns='symbol', values='close').ffill()
    caps = df.pivot(index='date', columns='symbol', values='market_cap').ffill().fillna(0.0)

    # 2. Returns & Log Returns
    returns = closes.pct_change().fillna(0.0)  # (T, N)
    log_returns = np.log(closes / closes.shift(1)).fillna(0.0)

    # 3. Dynamic Graph Generation
    print("[Preprocess] Generating Dynamic Correlation Graphs (this may take time)...")

    data_values = log_returns.values  # (T, N)
    T, N = data_values.shape

    # Convert to Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # (T, N)
    X_gpu = torch.tensor(data_values, device=device, dtype=torch.float32)

    # We want windows of shape (Num_Windows, N, Lags)
    # unfold(dimension, size, step) -> (dim, ..., size) appended at end
    # X_gpu: (T, N)
    # unfold(0, lags, 1) -> (Num_Windows, N, Lags)
    windows = X_gpu.unfold(0, lags, 1)  # Shape: (T-lags+1, N, Lags)

    # Standardize data within each window (along Lags dimension)
    # mean, std: (Num_Windows, N, 1)
    mean = windows.mean(dim=2, keepdim=True)
    std = windows.std(dim=2, keepdim=True) + 1e-9
    windows_std = (windows - mean) / std  # (Num_Windows, N, Lags)

    # Correlation Matrix Calculation: (Z @ Z.T) / (Lags - 1)
    # We want correlation between Stocks (N x N) for each window.
    # Input: (B, N, L). Matmul: (B, N, L) @ (B, L, N) -> (B, N, N)
    corr_matrices = torch.bmm(windows_std, windows_std.transpose(1, 2)) / (lags - 1)

    # Thresholding
    THRESHOLD = 0.6
    pos_mask = (corr_matrices > THRESHOLD).float()
    neg_mask = (corr_matrices < -THRESHOLD).float()

    # Remove self-loops (Diagonal)
    # diag_mask: (1, N, N) -> broadcasting to (B, N, N) works
    diag_mask = torch.eye(N, device=device).unsqueeze(0)
    pos_mask = pos_mask * (1 - diag_mask)
    neg_mask = neg_mask * (1 - diag_mask)  # Negative correlation doesn't have self-loop usually, but good for safety

    # Move to CPU
    adj_pos_all = pos_mask.cpu()
    adj_neg_all = neg_mask.cpu()

    # Data Dictionary
    X_tensor = torch.tensor(data_values, dtype=torch.float32).unsqueeze(-1)  # (T, N, 1)
    Y_tensor = torch.tensor(returns.values, dtype=torch.float32)
    Cap_tensor = torch.tensor(caps.values, dtype=torch.float32)

    data = {
        "X": X_tensor,
        "Y": Y_tensor,
        "Caps": Cap_tensor,
        "Adj_Pos": adj_pos_all,  # (T-L+1, N, N)
        "Adj_Neg": adj_neg_all,  # (T-L+1, N, N)
        "dates": closes.index,
        "symbols": closes.columns.tolist()
    }

    with open(cache_path, "wb") as f: pickle.dump(data, f)
    print(f"[Preprocess] Saved to {cache_path}")
    return data


# -----------------------------------------------------------------------------
# 2. THGNN Model Architecture
# -----------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT Layer for Temporal Graph Attention
    Eq (5) & (6)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (B, N, in_features)
        # adj: (B, N, N)
        Wh = torch.matmul(h, self.W)  # (B, N, out)

        # Self-attention on the nodes - Shared attention mechanism
        # Prepare for broadcasting: (B, N, 1, out) and (B, 1, N, out)
        B, N, _ = Wh.size()
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)

        # Concat -> (B, N, N, 2*out)
        a_input = torch.cat([Wh1, Wh2], dim=-1)

        # (B, N, N, 1) -> (B, N, N)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (B, N, out)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class HeterogeneousGraphAttention(nn.Module):
    """
    Heterogeneous Graph Attention Mechanism
    Aggregates H_self, H_pos, H_neg
    Eq (7), (8), (9)
    """

    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, attn_dim)
        self.q = nn.Parameter(torch.Tensor(attn_dim, 1))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)

    def forward(self, h_self, h_pos, h_neg):
        # Stack inputs: (B, N, 3, D)
        stack = torch.stack([h_self, h_pos, h_neg], dim=2)

        # Calculate Importance w_r
        # (B, N, 3, D) -> (B, N, 3, attn_dim)
        u = torch.tanh(self.W(stack))
        # (B, N, 3, attn_dim) @ (attn_dim, 1) -> (B, N, 3, 1)
        attn_scores = torch.matmul(u, self.q)

        # Average importance over all nodes (as per Eq 8 text description "average... importance of each company relation")
        # Or per-node attention? The formula says sum over V. Let's do per-node for better granularity, or global if strictly following text.
        # "we average the importance of all node embeddings" -> Global weights per relation type.
        # w_r = (1/|V|) * Sum(q^T * tanh(...))
        w_r = torch.mean(attn_scores, dim=1, keepdim=True)  # (B, 1, 3, 1)

        # Softmax -> beta
        beta = F.softmax(w_r, dim=2)  # (B, 1, 3, 1)

        # Weighted Sum
        # (B, N, 3, D) * (B, 1, 3, 1) -> sum dim 2 -> (B, N, D)
        out = torch.sum(stack * beta, dim=2)
        return out


class THGNN(nn.Module):
    """
    Full THGNN Architecture
    """

    def __init__(self, d_feat=1, d_in=64, d_enc=64, d_hid=64, d_att=64, n_heads=4, dropout=0.1):
        super().__init__()

        # 1. Historical Price Encoding
        self.input_proj = nn.Linear(d_feat, d_in)
        # Positional Encoding is often implicit in Transformers or added explicitly.
        # We use a standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=n_heads, dim_feedforward=d_hid, batch_first=True,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj_enc = nn.Linear(d_in, d_enc)

        # 2. Temporal Graph Attention
        # Two GATs for Pos and Neg relations
        self.gat_pos = GraphAttentionLayer(d_enc, d_att, dropout=dropout, alpha=0.2, concat=True)
        self.gat_neg = GraphAttentionLayer(d_enc, d_att, dropout=dropout, alpha=0.2, concat=True)

        # Self transformation
        self.linear_self = nn.Linear(d_enc, d_att)

        # 3. Heterogeneous Graph Attention
        self.hetero_attn = HeterogeneousGraphAttention(d_att, d_att)

        # 4. Prediction Head
        # Output is regression (score)
        self.head = nn.Sequential(
            nn.Linear(d_att, d_att // 2),
            nn.LeakyReLU(),
            nn.Linear(d_att // 2, 1)
        )

    def forward(self, x, adj_pos, adj_neg):
        # x: (B, N, T, F) -> Batch of (N stocks, T time, F feat)
        # Note: Previous MASTER code used (B, N, T, F) layout effectively.
        B, N, T, F = x.shape

        # Flatten for Transformer: Process each stock's time series independently
        # (B*N, T, F)
        x_flat = x.reshape(B * N, T, F)
        x_emb = self.input_proj(x_flat)

        # Transformer Encoding
        # Output: (B*N, T, d_in)
        h_enc_seq = self.transformer_encoder(x_emb)

        # Take last timestamp or pool? Paper implies using the sequence embedding.
        # Eq (2) suggests output is H_enc^t. Usually last step is used for prediction at t.
        h_enc = h_enc_seq[:, -1, :]  # (B*N, d_in)
        h_enc = self.proj_enc(h_enc)  # (B*N, d_enc)

        # Reshape for Graph: (B, N, d_enc)
        h_nodes = h_enc.reshape(B, N, -1)

        # Self Embedding
        h_self = self.linear_self(h_nodes)

        # Graph Attention
        h_pos = self.gat_pos(h_nodes, adj_pos)
        h_neg = self.gat_neg(h_nodes, adj_neg)

        # Heterogeneous Aggregation
        z = self.hetero_attn(h_self, h_pos, h_neg)  # (B, N, d_att)

        # Prediction
        out = self.head(z)  # (B, N, 1)
        return out.squeeze(-1)


# -----------------------------------------------------------------------------
# 3. Experiment Runner & Metrics
# -----------------------------------------------------------------------------

def backtest_stats(returns_vec):
    """ASR, RMDD, AVol"""
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
        if k == 0:
            metrics[p] = 0.0;
            continue
        top_k = sorted_idx[:k]
        w = caps[top_k]
        err = (pred[top_k] - true[top_k]) ** 2
        metrics[p] = np.sum(w * err) / (np.sum(w) + 1e-12)
    return metrics


def prepare_windows(data, lags):
    X, Y, Caps = data['X'], data['Y'], data['Caps']
    Adj_Pos_All, Adj_Neg_All = data['Adj_Pos'], data['Adj_Neg']

    T_total = X.shape[0]
    X_l, Y_l, C_l, AP_l, AN_l = [], [], [], [], []

    # Valid range: lags to T_total
    # Graph data length is T_total - lags + 1
    # If t starts at lags, graph index is t - lags.

    for t in range(lags, T_total):
        X_l.append(X[t - lags: t])
        Y_l.append(Y[t])
        C_l.append(Caps[t - 1])

        # Graph at time t (decision time)
        # The pre-calculated Adjacency list starts from window [0:lags], which corresponds to t=lags.
        # So index is t - lags.
        idx = t - lags
        AP_l.append(Adj_Pos_All[idx])
        AN_l.append(Adj_Neg_All[idx])

    return (torch.stack(X_l), torch.stack(Y_l), torch.stack(C_l),
            torch.stack(AP_l), torch.stack(AN_l), data['dates'][lags:])


def run_single_seed(args, seed, cached_data):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, Caps, AP, AN, dates = prepare_windows(cached_data, args.lags)

    train_from = pd.to_datetime(args.train_from)
    valid_from = pd.to_datetime(args.valid_from)
    test_from = pd.to_datetime(args.test_from)
    test_to = pd.to_datetime(args.test_to)

    train_mask = (dates >= train_from) & (dates < valid_from)
    val_mask = (dates >= valid_from) & (dates < test_from)
    test_mask = (dates >= test_from) & (dates <= test_to)

    if not train_mask.any(): return None

    # Z-Score Normalization
    def get_z(y):
        mu = y.mean(dim=1, keepdim=True)
        sig = y.std(dim=1, keepdim=True) + 1e-6
        return (y - mu) / sig

    Y_train_z = get_z(Y[train_mask])
    Y_val_z = get_z(Y[val_mask])

    # Dataset includes Graphs
    train_ds = TensorDataset(X[train_mask], Y_train_z, AP[train_mask], AN[train_mask])
    val_ds = TensorDataset(X[val_mask], Y_val_z, AP[val_mask], AN[val_mask])
    test_ds = TensorDataset(X[test_mask], Y[test_mask], Caps[test_mask], AP[test_mask], AN[test_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # THGNN Model Setup
    model = THGNN(d_feat=X.shape[3]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None

    # Train
    for _ in range(args.epochs):
        model.train()
        for bx, by, bap, ban in train_loader:
            bx, by = bx.to(device), by.to(device)
            bap, ban = bap.to(device), ban.to(device)

            optimizer.zero_grad()
            # Input X needs to be (B, N, T, F)
            # Currently (B, T, N, F). Permute to (B, N, T, F)
            bx_perm = bx.permute(0, 2, 1, 3)

            loss = criterion(model(bx_perm, bap, ban), by)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx, by, bap, ban in val_loader:
                bx, by = bx.to(device), by.to(device)
                bap, ban = bap.to(device), ban.to(device)
                bx_perm = bx.permute(0, 2, 1, 3)
                vloss += criterion(model(bx_perm, bap, ban), by).item()

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
        for bx, by_raw, bcap, bap, ban in test_loader:
            bx = bx.to(device)
            bap, ban = bap.to(device), ban.to(device)
            bx_perm = bx.permute(0, 2, 1, 3)

            pred_z = model(bx_perm, bap, ban).cpu().numpy()
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

    data = build_thgnn_dataset(reg_code, args.lags)

    results = []
    print(f"Running {args.n_seeds} seeds for {reg_code} (THGNN)...")

    for s in range(args.n_seeds):
        res = run_single_seed(args, s, data)
        if res:
            results.append(res)
            print(f"Seed {s}: ASR={res['ASR']:.4f} | MSE={res['MSE']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (THGNN)")
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