import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
# 1. Independent Preprocessing for MASTER
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


def build_master_dataset(region, lags=5, cache_dir="cache_master"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"master_data_{region}_l{lags}.pkl")

    if os.path.exists(cache_path):
        print(f"[Preprocess] Loading cached MASTER dataset: {cache_path}")
        with open(cache_path, "rb") as f: return pickle.load(f)

    print(f"[Preprocess] Building new MASTER dataset for {region}...")
    df = fetch_price_data(region)

    # Pivot
    closes = df.pivot(index='date', columns='symbol', values='close').ffill()
    caps = df.pivot(index='date', columns='symbol', values='market_cap').ffill().fillna(0.0)

    # Returns (Target: t+1 Return)
    returns = closes.pct_change().fillna(0.0)  # (T, N)
    log_returns = np.log(closes / closes.shift(1)).fillna(0.0)

    # Tensor
    X_tensor = torch.tensor(log_returns.values, dtype=torch.float32).unsqueeze(-1)  # (T, N, 1)
    Y_tensor = torch.tensor(returns.values, dtype=torch.float32)  # (T, N)
    Cap_tensor = torch.tensor(caps.values, dtype=torch.float32)  # (T, N)

    data = {
        "X": X_tensor, "Y": Y_tensor, "Caps": Cap_tensor,
        "dates": closes.index, "symbols": closes.columns.tolist()
    }

    with open(cache_path, "wb") as f: pickle.dump(data, f)
    print(f"[Preprocess] Saved to {cache_path}")
    return data


# -----------------------------------------------------------------------------
# 2. MASTER Model
# -----------------------------------------------------------------------------

class MASTER(nn.Module):
    def __init__(self, d_feat=1, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        self.intra_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.inter_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LeakyReLU(), nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        B, T, N, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        x_emb = self.input_proj(x_flat)
        out_intra, _ = self.intra_rnn(x_emb)
        h = self.ln1(out_intra).reshape(B, N, T, -1).permute(0, 2, 1, 3)

        h_flat = h.reshape(B * T, N, -1)
        out_inter, _ = self.inter_attn(h_flat, h_flat, h_flat)
        h_inter = self.ln2(h_flat + out_inter).reshape(B, T, N, -1)

        return self.head(h_inter[:, -1, :, :]).squeeze(-1)


# -----------------------------------------------------------------------------
# 3. Experiment Runner & Metrics
# -----------------------------------------------------------------------------

def backtest_stats(returns_vec):
    """
    Calculate Investment Metrics: ASR, RMDD, AVol
    returns_vec: List of daily portfolio returns (Scalar, Raw Scale)
    """
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
    """
    CWMSE calculation on Percentage Scale
    """
    sorted_idx = np.argsort(caps)[::-1]
    metrics = {}
    for p in p_list:
        k = min(p, len(sorted_idx))
        if k == 0:
            metrics[p] = 0.0;
            continue

        top_k = sorted_idx[:k]
        w = caps[top_k]
        err = (pred[top_k] - true[top_k]) ** 2  # pred, true are already in % scale
        metrics[p] = np.sum(w * err) / (np.sum(w) + 1e-12)
    return metrics


def prepare_windows(data, lags):
    X, Y, Caps = data['X'], data['Y'], data['Caps']
    T_total = X.shape[0]
    X_l, Y_l, C_l = [], [], []

    for i in range(lags, T_total):
        X_l.append(X[i - lags: i])
        Y_l.append(Y[i])
        C_l.append(Caps[i - 1])

    return torch.stack(X_l), torch.stack(Y_l), torch.stack(C_l), data['dates'][lags:]


def run_single_seed(args, seed, cached_data):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, Caps, dates = prepare_windows(cached_data, args.lags)

    train_from = pd.to_datetime(args.train_from)
    valid_from = pd.to_datetime(args.valid_from)
    test_from = pd.to_datetime(args.test_from)
    test_to = pd.to_datetime(args.test_to)

    train_mask = (dates >= train_from) & (dates < valid_from)
    val_mask = (dates >= valid_from) & (dates < test_from)
    test_mask = (dates >= test_from) & (dates <= test_to)

    if not train_mask.any(): return None

    # Z-Score for Training
    def get_z(y):
        mu = y.mean(dim=1, keepdim=True)
        sig = y.std(dim=1, keepdim=True) + 1e-6
        return (y - mu) / sig

    Y_train_z = get_z(Y[train_mask])
    Y_val_z = get_z(Y[val_mask])

    train_ds = TensorDataset(X[train_mask], Y_train_z)
    val_ds = TensorDataset(X[val_mask], Y_val_z)
    test_ds = TensorDataset(X[test_mask], Y[test_mask], Caps[test_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MASTER(d_feat=X.shape[3], d_model=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None

    for _ in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vloss += criterion(model(bx), by).item()

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_weights = model.state_dict().copy()

    # Test
    model.load_state_dict(best_weights)
    model.eval()

    cwmse_sums = {3: 0, 5: 0, 10: 0}
    daily_rets = []
    mse_sum, mae_sum, count = 0, 0, 0

    # [SCALING FACTOR] Convert to Percentage (e.g., 0.01 -> 1.0) for readable MSE/MAE
    SCALE = 100.0

    with torch.no_grad():
        for bx, by_raw, bcap in test_loader:
            bx = bx.to(device)
            pred_z = model(bx).cpu().numpy()
            true_raw = by_raw.numpy()
            caps_raw = bcap.numpy()

            for i in range(len(pred_z)):
                # 1. Denormalize to Raw Scale (Decimals)
                mu = true_raw[i].mean()
                sig = true_raw[i].std() + 1e-6
                pred_raw = pred_z[i] * sig + mu

                # 2. Investment Metrics (Use RAW Decimals for Compounding)
                top5_idx = np.argsort(pred_z[i])[-5:]
                daily_rets.append(np.mean(true_raw[i][top5_idx]))

                # 3. Error Metrics (Use PERCENTAGE Scale for Readability)
                pred_pct = pred_raw * SCALE
                true_pct = true_raw[i] * SCALE

                # MSE / MAE
                mse_sum += np.mean((pred_pct - true_pct) ** 2)
                mae_sum += np.mean(np.abs(pred_pct - true_pct))

                # CWMSE
                c_res = calc_cwmse(pred_pct, true_pct, caps_raw[i])
                for k in cwmse_sums: cwmse_sums[k] += c_res[k]

                count += 1

    # Aggregate Metrics
    metrics = {}
    metrics['MSE'] = mse_sum / count
    metrics['MAE'] = mae_sum / count
    metrics['RMSE'] = np.sqrt(metrics['MSE'])  # Added RMSE for better readability

    for k, v in cwmse_sums.items():
        metrics[f'CWMSE@{k}'] = v / count

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

    data = build_master_dataset(reg_code, args.lags)

    results = []
    print(f"Running {args.n_seeds} seeds for {reg_code}...")

    for s in range(args.n_seeds):
        res = run_single_seed(args, s, data)
        if res:
            results.append(res)
            print(f"Seed {s}: ASR={res['ASR']:.4f} | RMDD={res['RMDD']:.4f} | MSE(%)={res['MSE']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (Error Metrics in % Scale)")
    print("-" * 80)
    df = pd.DataFrame(results)

    # Full list of metrics to report
    cols = ['ASR', 'RMDD', 'AVol', 'MSE', 'MAE', 'RMSE', 'CWMSE@3', 'CWMSE@5', 'CWMSE@10']
    for col in cols:
        if col in df.columns:
            mean_v = df[col].mean()
            std_v = df[col].std()
            print(f"{col:<10} | {mean_v:.4f} ± {std_v:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()