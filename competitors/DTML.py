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
# 1. Independent Preprocessing for DTML
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

    if 'adjusted_close' not in df.columns:
        df['adjusted_close'] = df['close']

    return df.sort_values(['symbol', 'date'])


def build_dtml_dataset(region, lags=20, cache_dir="cache_dtml"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"dtml_data_{region}_l{lags}.pkl")

    if os.path.exists(cache_path):
        print(f"[Preprocess] Loading cached DTML dataset: {cache_path}")
        with open(cache_path, "rb") as f: return pickle.load(f)

    print(f"[Preprocess] Building new DTML dataset for {region}...")
    df = fetch_price_data(region)

    opens = df.pivot(index='date', columns='symbol', values='open').ffill()
    highs = df.pivot(index='date', columns='symbol', values='high').ffill()
    lows = df.pivot(index='date', columns='symbol', values='low').ffill()
    closes = df.pivot(index='date', columns='symbol', values='close').ffill()
    adj_closes = df.pivot(index='date', columns='symbol', values='adjusted_close').ffill()
    caps = df.pivot(index='date', columns='symbol', values='market_cap').ffill().fillna(0.0)

    if adj_closes.isnull().all().all():
        print("[Warning] Adjusted Close not found, using Close instead.")
        adj_closes = closes

    z_open = opens / closes - 1
    z_high = highs / closes - 1
    z_low = lows / closes - 1
    z_close = closes / closes.shift(1) - 1
    z_adj_close = adj_closes / adj_closes.shift(1) - 1

    long_term_features = []
    for k in [5, 10, 15, 20, 25, 30]:
        ma_k = adj_closes.rolling(window=k).mean()
        z_dk = (ma_k / adj_closes) - 1
        long_term_features.append(z_dk)

    feature_list = [z_open, z_high, z_low, z_close, z_adj_close] + long_term_features
    features_np = np.stack([f.fillna(0.0).values for f in feature_list], axis=2)

    macro_features_np = np.mean(features_np, axis=1)

    next_returns = z_close.shift(-1).fillna(0.0)
    targets = (next_returns > 0).astype(int).values
    raw_returns = next_returns.values
    caps_values = caps.values

    X_stock_wins = []
    X_macro_wins = []
    Y_wins = []
    R_wins = []
    C_wins = []

    T_total, N, F = features_np.shape

    for i in range(lags, T_total - 1):
        X_stock_wins.append(features_np[i - lags:i])
        X_macro_wins.append(macro_features_np[i - lags:i])
        Y_wins.append(targets[i])
        R_wins.append(raw_returns[i])
        C_wins.append(caps_values[i])  # Use Cap at t for weighting

    X_s = np.array(X_stock_wins).transpose(0, 2, 1, 3)
    X_m = np.array(X_macro_wins)
    Y = np.array(Y_wins)
    R = np.array(R_wins)
    C = np.array(C_wins)

    data = {
        "X_stock": torch.tensor(X_s, dtype=torch.float32),
        "X_macro": torch.tensor(X_m, dtype=torch.float32),
        "Y": torch.tensor(Y, dtype=torch.long),
        "R": torch.tensor(R, dtype=torch.float32),
        "C": torch.tensor(C, dtype=torch.float32),
        "dates": closes.index[lags:-1]
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Preprocess] Saved to {cache_path}")
    return data


# -----------------------------------------------------------------------------
# 2. DTML Model Architecture
# -----------------------------------------------------------------------------

class AttLstm(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda=True):
        super(self.__class__, self).__init__()
        self.lstm_hidden_layer = hidden_size
        self.use_cuda = use_cuda
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, _input):
        batch_size = _input.size(0)
        device = _input.device
        h_state = torch.zeros((batch_size, self.lstm_hidden_layer), device=device)
        c_state = torch.zeros((batch_size, self.lstm_hidden_layer), device=device)
        h_states = []
        for j in range(_input.size(1)):
            h_state, c_state = self.lstm_cell(_input[:, j, :], (h_state, c_state))
            h_states.append(h_state.unsqueeze(1))
        h_states = torch.cat(h_states, dim=1)
        att_score = torch.matmul(h_states, h_state.unsqueeze(2))
        att_dist = F.softmax(att_score, dim=1)
        context_vector = torch.matmul(att_dist.transpose(1, 2), h_states)
        return context_vector.squeeze(1)


class Dtml(nn.Module):
    def __init__(self, n_stock_input_vars, n_macro_input_vars, n_stock, n_time, n_heads, d_lstm_input=None,
                 lstm_hidden_layer=64, use_cuda=True):
        super(self.__class__, self).__init__()
        if d_lstm_input is None: d_lstm_input = n_stock_input_vars
        self.lstm_hidden_layer = lstm_hidden_layer
        self.use_cuda = use_cuda

        self.stock_f_tr_layer = nn.Sequential(nn.Linear(n_stock_input_vars, d_lstm_input), nn.Tanh())
        self.macro_f_tr_layer = nn.Sequential(nn.Linear(n_macro_input_vars, d_lstm_input), nn.Tanh())

        self.stock_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, use_cuda=use_cuda)
        self.macro_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, use_cuda=use_cuda)

        self.norm_weight = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))
        self.norm_bias = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))
        self.macro_weight = nn.Parameter(torch.randn(1))

        self.multi_head_att = nn.MultiheadAttention(lstm_hidden_layer, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_layer, lstm_hidden_layer * 4), nn.ReLU(),
            nn.Linear(lstm_hidden_layer * 4, lstm_hidden_layer)
        )
        self.final_layer = nn.Linear(lstm_hidden_layer, 2)  # Classification Output

        # Modification for Regression to enable MSE/CWMSE reporting:
        self.final_layer_reg = nn.Linear(lstm_hidden_layer, 1)

    def forward(self, stock_input, macro_input):
        batch_size, n_stock, n_time, n_feat = stock_input.size()
        stock_input_flat = stock_input.view(-1, n_time, n_feat)
        stock_emb = self.stock_f_tr_layer(stock_input_flat)
        macro_emb = self.macro_f_tr_layer(macro_input)

        c_matrix = self.stock_att_lstm(stock_emb).view(batch_size, n_stock, -1)
        macro_context = self.macro_att_lstm(macro_emb)

        mean = torch.mean(c_matrix, dim=(1, 2), keepdim=True)
        std = torch.std(c_matrix, dim=(1, 2), keepdim=True) + 1e-9
        c_norm = (c_matrix - mean) / std
        c_matrix = self.norm_weight.unsqueeze(0) * c_norm + self.norm_bias.unsqueeze(0)

        ml_c_matrix = c_matrix + self.macro_weight * macro_context.unsqueeze(1)
        att_value_matrix, _ = self.multi_head_att(ml_c_matrix, ml_c_matrix, ml_c_matrix)

        res1 = ml_c_matrix + att_value_matrix
        mlp_out = self.mlp(res1)
        out_matrix = torch.tanh(res1 + mlp_out)

        # Regression Output
        return self.final_layer_reg(out_matrix).squeeze(-1)  # (Batch, Stock)


# -----------------------------------------------------------------------------
# 3. Experiment Runner
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


def run_single_seed(args, seed, cached_data):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_s, X_m, Y_cls, R, Caps = cached_data['X_stock'], cached_data['X_macro'], cached_data['Y'], cached_data['R'], \
                               cached_data['C']
    dates = cached_data['dates']

    def get_z(y):
        mu = y.mean(dim=1, keepdim=True)
        sig = y.std(dim=1, keepdim=True) + 1e-6
        return (y - mu) / sig

    train_from = pd.to_datetime(args.train_from)
    valid_from = pd.to_datetime(args.valid_from)
    test_from = pd.to_datetime(args.test_from)
    test_to = pd.to_datetime(args.test_to)

    train_mask = (dates >= train_from) & (dates < valid_from)
    val_mask = (dates >= valid_from) & (dates < test_from)
    test_mask = (dates >= test_from) & (dates <= test_to)

    if not train_mask.any(): return None

    Y_train_z = get_z(R[train_mask])
    Y_val_z = get_z(R[val_mask])

    train_ds = TensorDataset(X_s[train_mask], X_m[train_mask], Y_train_z)
    val_ds = TensorDataset(X_s[val_mask], X_m[val_mask], Y_val_z)
    test_ds = TensorDataset(X_s[test_mask], X_m[test_mask], R[test_mask], Caps[test_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    n_stock = X_s.shape[1]
    n_time = X_s.shape[2]
    n_feat = X_s.shape[3]

    model = Dtml(
        n_stock_input_vars=n_feat, n_macro_input_vars=n_feat,
        n_stock=n_stock, n_time=n_time, n_heads=4,
        d_lstm_input=64, lstm_hidden_layer=64, use_cuda=torch.cuda.is_available()
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None

    for ep in range(args.epochs):
        model.train()
        for bx_s, bx_m, by in train_loader:
            bx_s, bx_m, by = bx_s.to(device), bx_m.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx_s, bx_m)  # (B, S)
            loss = criterion(pred, by)

            reg_loss = 0
            for name, param in model.named_parameters():
                if 'final_layer' in name:
                    reg_loss += torch.norm(param, 2)
            loss += 1.0 * reg_loss

            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for bx_s, bx_m, by in val_loader:
                bx_s, bx_m, by = bx_s.to(device), bx_m.to(device), by.to(device)
                pred = model(bx_s, bx_m)
                vloss += criterion(pred, by).item()

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_weights = model.state_dict().copy()

    model.load_state_dict(best_weights)
    model.eval()

    cwmse_sums = {3: 0, 5: 0, 10: 0}
    daily_rets = []
    mse_sum, mae_sum, count = 0, 0, 0
    SCALE = 100.0

    with torch.no_grad():
        for bx_s, bx_m, by_raw, bcap in test_loader:
            bx_s, bx_m = bx_s.to(device), bx_m.to(device)
            pred_z = model(bx_s, bx_m).cpu().numpy()
            true_raw = by_raw.numpy()
            caps_raw = bcap.numpy()

            for i in range(len(pred_z)):
                mu = true_raw[i].mean()
                sig = true_raw[i].std() + 1e-6
                pred_raw = pred_z[i] * sig + mu

                top_k_idx = np.argsort(pred_raw)[-5:]
                daily_rets.append(np.mean(true_raw[i][top_k_idx]))

                pred_pct = pred_raw * SCALE
                true_pct = true_raw[i] * SCALE

                mse_sum += np.mean((pred_pct - true_pct) ** 2)
                mae_sum += np.mean(np.abs(pred_pct - true_pct))

                c_res = calc_cwmse(pred_pct, true_pct, caps_raw[i])
                for k in cwmse_sums: cwmse_sums[k] += c_res[k]

                count += 1

    metrics = {}
    metrics['MSE'] = mse_sum / count
    metrics['MAE'] = mae_sum / count
    metrics['RMSE'] = np.sqrt(metrics['MSE'])

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

    data = build_dtml_dataset(reg_code, args.lags)

    results = []
    print(f"Running {args.n_seeds} seeds for {reg_code} (DTML-Regression)...")

    for s in range(args.n_seeds):
        res = run_single_seed(args, s, data)
        if res:
            results.append(res)
            print(f"Seed {s}: ASR={res['ASR']:.4f} | RMDD={res['RMDD']:.4f} | MSE(%)={res['MSE']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (DTML with Regression Head)")
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