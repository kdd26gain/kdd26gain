import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# 1. Shared Preprocessing (Sequential Data)
# -----------------------------------------------------------------------------

def fetch_price_data(region, start_date="2013-01-01", end_date="2025-03-31"):
    """
    Assumes data is located at '../data/stocks_{region}.csv' relative to this script.
    """
    # Define data path (Parent directory's data folder)
    data_dir = "../data"
    csv_file = os.path.join(data_dir, f"stocks_{region}.csv")

    if not os.path.exists(csv_file):
        # Fallback: Check current directory or raise error
        if os.path.exists(f"data/stocks_{region}.csv"):
            csv_file = f"data/stocks_{region}.csv"
        else:
            raise FileNotFoundError(f"Data file not found: {csv_file}. Please ensure CSV files are in '../data/'.")

    print(f"[Preprocess] Loading raw data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Ensure Date format
    df['date'] = pd.to_datetime(df['date'])

    # Filter by Date
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    # Region-specific filtering (Logic transferred from preprocess.py)
    if region == "KOR":
        df = df[df['symbol'] != '010130']
    if region == "CHN":
        df = df[df['symbol'] != 'SZSE: 002714']

    # Ensure required columns exist
    required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'market_cap']
    # Check if adjusted_close exists, if not create it from close (fallback)
    if 'adjusted_close' not in df.columns:
        df['adjusted_close'] = df['close']

    return df.sort_values(['symbol', 'date'])


def build_sequential_dataset(region, lags=20, cache_dir="cache_lstm"):
    """
    Generate standard sequential features for LSTM/ALSTM.
    Shared by both models. Includes Caps for CWMSE.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"seq_data_{region}_l{lags}.pkl")

    if os.path.exists(cache_path):
        print(f"[Preprocess] Loading cached dataset: {cache_path}")
        with open(cache_path, "rb") as f: return pickle.load(f)

    print(f"[Preprocess] Building new Sequential dataset for {region}...")
    df = fetch_price_data(region)

    # Pivot Data
    # Using ffill() to handle missing days
    opens = df.pivot(index='date', columns='symbol', values='open').ffill()
    highs = df.pivot(index='date', columns='symbol', values='high').ffill()
    lows = df.pivot(index='date', columns='symbol', values='low').ffill()
    closes = df.pivot(index='date', columns='symbol', values='close').ffill()
    adj_closes = df.pivot(index='date', columns='symbol', values='adjusted_close').ffill()
    caps = df.pivot(index='date', columns='symbol', values='market_cap').ffill().fillna(0.0)

    # Fallback if adjusted_close is all NaN
    if adj_closes.isnull().all().all():
        adj_closes = closes

    # Feature Engineering
    z_open = opens / closes - 1
    z_high = highs / closes - 1
    z_low = lows / closes - 1
    z_close = closes / closes.shift(1) - 1
    z_adj = adj_closes / adj_closes.shift(1) - 1

    feats = [z_open, z_high, z_low, z_close, z_adj]
    for k in [5, 10, 20]:
        ma = adj_closes.rolling(k).mean()
        feats.append((ma / adj_closes) - 1)

    # Stack features: (Time, N, F)
    features_np = np.stack([f.fillna(0.0).values for f in feats], axis=2)

    # Targets: Regression on Returns (Next Day)
    next_returns = z_close.shift(-1).fillna(0.0)
    raw_returns = next_returns.values
    caps_values = caps.values

    X_wins, Y_wins, R_wins, C_wins = [], [], [], []
    T_total, N, F = features_np.shape

    # Create Sliding Windows
    for i in range(lags, T_total - 1):
        X_wins.append(features_np[i - lags:i])  # (Lags, N, F)
        Y_wins.append(raw_returns[i])  # (N,)
        R_wins.append(raw_returns[i])  # (N,)
        C_wins.append(caps_values[i])  # (N,)

    X_out = np.array(X_wins).transpose(0, 2, 1, 3)  # (B, N, L, F)

    data = {
        "X": torch.tensor(X_out, dtype=torch.float32),
        "Y": torch.tensor(np.array(Y_wins), dtype=torch.float32),  # Float for regression
        "R": torch.tensor(np.array(R_wins), dtype=torch.float32),
        "C": torch.tensor(np.array(C_wins), dtype=torch.float32),
        "dates": closes.index[lags:-1]
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Preprocess] Saved to {cache_path}")
    return data


# -----------------------------------------------------------------------------
# 2. Models: LSTM & ALSTM
# -----------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        pass

    def forward(self, lstm_output, last_hidden):
        scores = torch.bmm(lstm_output, last_hidden.unsqueeze(2))
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights, dim=1)
        return context


class StockRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

        if self.use_attention:
            self.attention = TemporalAttention(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression Output
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)

        if self.use_attention:
            last_hidden = hn[-1]
            feat_vec = self.attention(output, last_hidden)
        else:
            feat_vec = hn[-1]

        return self.fc(feat_vec)  # (B, 1)


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
        if k == 0:
            metrics[p] = 0.0
            continue
        top_k = sorted_idx[:k]
        w = caps[top_k]
        err = (pred[top_k] - true[top_k]) ** 2
        metrics[p] = np.sum(w * err) / (np.sum(w) + 1e-12)
    return metrics


def run_model(args, seed, cached_data, model_type="LSTM"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, R, Caps = cached_data['X'], cached_data['Y'], cached_data['R'], cached_data['C']
    dates = cached_data['dates']

    # Z-Score for targets (Regression Stability)
    def get_z(y):
        mu = y.mean()
        sig = y.std() + 1e-6
        return (y - mu) / sig

    # Convert args dates to datetime for comparison
    train_from = pd.to_datetime(args.train_from)
    valid_from = pd.to_datetime(args.valid_from)
    test_from = pd.to_datetime(args.test_from)
    test_to = pd.to_datetime(args.test_to)

    train_mask = (dates >= train_from) & (dates < valid_from)
    val_mask = (dates >= valid_from) & (dates < test_from)
    test_mask = (dates >= test_from) & (dates <= test_to)

    if not train_mask.any():
        print(f"[Warning] No training data found for {args.region}. Check dates.")
        return None, None, None, None

    # Prepare Flat Loaders for Training (Stock Independent)
    def prepare_loader(mask, shuffle=False):
        X_sub = X[mask].reshape(-1, X.shape[2], X.shape[3])  # (Times*N, L, F)
        # Use normalized targets for training
        Y_norm = get_z(Y[mask]).reshape(-1)
        return DataLoader(TensorDataset(X_sub, Y_norm), batch_size=args.batch_size * 32, shuffle=shuffle)

    train_loader = prepare_loader(train_mask, shuffle=True)
    val_loader = prepare_loader(val_mask, shuffle=False)

    # Prepare Test Loader (Structured for Portfolio/CWMSE)
    # X: (T, N, L, F), Y: (T, N), R: (T, N), C: (T, N)
    test_ds = TensorDataset(X[test_mask], Y[test_mask], R[test_mask], Caps[test_mask])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Get Test Dates
    test_dates_list = dates[test_mask]

    feat_dim = X.shape[3]
    use_att = (model_type == "ALSTM")
    model = StockRNN(input_dim=feat_dim, hidden_dim=128, use_attention=use_att).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None

    for ep in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx).squeeze(-1)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx).squeeze(-1)
                vloss += criterion(pred, by).item()

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_weights = model.state_dict().copy()

    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()

    cwmse_sums = {3: 0, 5: 0, 10: 0}
    daily_rets = []
    mse_sum, mae_sum, count = 0, 0, 0
    SCALE = 100.0

    # Store predictions for Ensemble
    full_preds_list = []
    full_trues_list = []

    with torch.no_grad():
        for bx_struct, by_struct, br_struct, bc_struct in test_loader:
            # bx_struct: (BatchTime, N, L, F)
            B_t, N, L, F_dim = bx_struct.shape
            bx_flat = bx_struct.reshape(-1, L, F_dim).to(device)  # (B*N, L, F)

            # Predict Z-score
            pred_z = model(bx_flat).squeeze(-1).cpu().numpy()  # (B*N,)

            # Reshape
            pred_z_matrix = pred_z.reshape(B_t, N)
            true_raw_matrix = br_struct.numpy()  # Raw Returns
            caps_matrix = bc_struct.numpy()

            for i in range(B_t):
                # Denormalize
                mu = true_raw_matrix[i].mean()
                sig = true_raw_matrix[i].std() + 1e-6
                pred_raw = pred_z_matrix[i] * sig + mu

                # Append for Ensemble
                full_preds_list.append(pred_raw)  # (N,)
                full_trues_list.append(true_raw_matrix[i])  # (N,)

                # Investment
                top_k_idx = np.argsort(pred_raw)[-5:]
                daily_rets.append(np.mean(true_raw_matrix[i][top_k_idx]))

                # Metrics
                pred_pct = pred_raw * SCALE
                true_pct = true_raw_matrix[i] * SCALE

                mse_sum += np.mean((pred_pct - true_pct) ** 2)
                mae_sum += np.mean(np.abs(pred_pct - true_pct))

                c_res = calc_cwmse(pred_pct, true_pct, caps_matrix[i])
                for k in cwmse_sums: cwmse_sums[k] += c_res[k]

                count += 1

    if count == 0:
        return None, None, None, None

    metrics = {}
    metrics['MSE'] = mse_sum / count
    metrics['MAE'] = mae_sum / count
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    for k, v in cwmse_sums.items():
        metrics[f'CWMSE@{k}'] = v / count

    metrics['ASR'], metrics['RMDD'], metrics['AVol'] = backtest_stats(daily_rets)

    # Stack collected arrays
    # pred_array: (Total_Time, N)
    pred_array = np.stack(full_preds_list, axis=0)
    true_array = np.stack(full_trues_list, axis=0)

    return metrics, pred_array, true_array, test_dates_list


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

    # Map Input Region to File Code (e.g. S&P500 -> USA)
    rmap = {"S&P500": "USA", "CSI300": "CHN", "EURO50": "EUR", "KP200": "KOR"}
    reg_code = rmap.get(args.region, args.region)

    data = build_sequential_dataset(reg_code, args.lags)

    print(f"Running {args.n_seeds} seeds for LSTM & ALSTM ({reg_code})...")

    # Store results
    summary_metrics = {"LSTM": [], "ALSTM": []}
    # Store predictions for Ensemble: list of (T, N) arrays
    preds_collection = {"LSTM": [], "ALSTM": []}

    # We need True Returns and Dates only once (they are same for all seeds/models)
    true_returns = None
    test_dates = None

    for s in range(args.n_seeds):
        # Run LSTM
        res_lstm, preds_l, trues_l, dates_l = run_model(args, s, data, "LSTM")
        if res_lstm:
            summary_metrics["LSTM"].append(res_lstm)
            preds_collection["LSTM"].append(preds_l)
            print(f"[Seed {s}] LSTM : ASR={res_lstm['ASR']:.4f} | MSE={res_lstm['MSE']:.4f}")

            if true_returns is None:
                true_returns = trues_l
                test_dates = dates_l

        # Run ALSTM
        res_alstm, preds_a, trues_a, dates_a = run_model(args, s, data, "ALSTM")
        if res_alstm:
            summary_metrics["ALSTM"].append(res_alstm)
            preds_collection["ALSTM"].append(preds_a)
            print(f"[Seed {s}] ALSTM: ASR={res_alstm['ASR']:.4f} | MSE={res_alstm['MSE']:.4f}")

    # -------------------------------------------------------
    # Report Average Metrics
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Metric':<10} | {'LSTM (Mean ± Std)':<35} | {'ALSTM (Mean ± Std)':<35}")
    print("-" * 100)

    metrics_list = ['ASR', 'RMDD', 'AVol', 'MSE', 'MAE', 'RMSE', 'CWMSE@3', 'CWMSE@5', 'CWMSE@10']

    for m in metrics_list:
        vals_l = [x[m] for x in summary_metrics["LSTM"]]
        if vals_l:
            mean_l, std_l = np.mean(vals_l), np.std(vals_l)
            str_l = f"{mean_l:.4f} ± {std_l:.4f}"
        else:
            str_l = "N/A"

        vals_a = [x[m] for x in summary_metrics["ALSTM"]]
        if vals_a:
            mean_a, std_a = np.mean(vals_a), np.std(vals_a)
            str_a = f"{mean_a:.4f} ± {std_a:.4f}"
        else:
            str_a = "N/A"

        print(f"{m:<10} | {str_l:<35} | {str_a:<35}")
    print("=" * 100)

    # -------------------------------------------------------
    # Ensemble & Save CSV
    # -------------------------------------------------------
    print("\n[Ensemble] Generating CSV files...")

    if test_dates is not None:
        dates_str = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in test_dates]

        for model_name in ["LSTM", "ALSTM"]:
            preds_list = preds_collection[model_name]
            if not preds_list: continue

            # Average Predictions across seeds: (Seeds, T, N) -> (T, N)
            ensemble_preds = np.mean(np.stack(preds_list), axis=0)

            # Calculate Daily Returns for Ensemble (Top-5 Long)
            ens_daily_rets = []
            for t in range(len(ensemble_preds)):
                pred_t = ensemble_preds[t]
                true_t = true_returns[t]
                top5_idx = np.argsort(pred_t)[-5:]
                ens_daily_rets.append(np.mean(true_t[top5_idx]))

            # Calculate Cumulative PV
            cum_pv = np.cumprod(1 + np.array(ens_daily_rets))

            # Create DataFrame
            df_ens = pd.DataFrame({
                "date": dates_str,
                "daily_return": ens_daily_rets,
                "cumulative_pv": cum_pv
            })

            # Save
            csv_name = f"Ensemble_{model_name}_{reg_code}.csv"
            df_ens.to_csv(csv_name, index=False)
            print(f"  -> Saved {csv_name}")


if __name__ == "__main__":
    main()