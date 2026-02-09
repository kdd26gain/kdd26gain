###############################################################################
# preprocess.py
# Loads raw CSV data from /data and converts it into Tensor datasets.
###############################################################################
import numpy as np
import pandas as pd
import torch
import pickle
import os
import time
from typing import Dict, List, Optional
from utils_path import dataset_path, get_data_file_path, _REGION_TAG

START_DATE, TEST_END = "2013-01-02", "2025-03-31"

# Macro variables configuration
RAW_MACRO = ["fx", "wti", "spread", "gold", "vix"]
NO_LOGRET = ["spread"]
LOGRET_COLS = [c for c in RAW_MACRO if c not in NO_LOGRET]
MACRO_COLS: List[str] = RAW_MACRO  # Updated in get_macro_data()


def save_dataset(*,
                 lags: int,
                 cutoff_date: str,
                 region: str,
                 macro: str = "USA",
                 use_index: bool = False,
                 **tensors) -> None:
    path = dataset_path(lags, cutoff_date, region, macro, use_index=use_index)
    with open(path, "wb") as f:
        pickle.dump(tensors, f)
    print(f"[Preprocess] Saved to {path}")
    print(f"  X_static: {tensors['X_static'].shape}")
    print(f"  F_label:  {tensors['F_label'].shape}")


# ════════════════════════════════════════════════════════════════════════════
# 1) Price Data Loading
# ════════════════════════════════════════════════════════════════════════════
def get_prices(region: str,
               start: str = START_DATE,
               end: str = TEST_END) -> pd.DataFrame:
    # Adjust start dates based on region availability
    if region.upper() == "KOR":
        start = "2015-01-01"
    elif region.upper() == "CHN":
        start = "2013-01-04"
    elif region.upper() in ["USA", "EUR"]:
        start = "2013-01-02"

    t0 = time.time()

    # Load from CSV instead of DB
    csv_file = get_data_file_path(f"stocks_{region}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Stock data file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])

    # Filter by date
    df = df[(df['date'] >= start) & (df['date'] <= end)]

    # Filter specific symbols for KOR/CHN datasets
    if region.upper() == "KOR":
        df = df[df['symbol'] != '010130']
    if region.upper() == "CHN":
        df = df[df['symbol'] != 'SZSE: 002714']

    print(f"[get_prices-{region}] Loaded {df.shape} rows ({time.time() - t0:.2f}s)")
    return df


def preprocess_data(df: pd.DataFrame, debug=False) -> pd.DataFrame:
    if df.empty:
        return df

    numeric = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric)

    grp = df.groupby('symbol')

    # Log returns (Close)
    df['prev_close'] = grp['close'].shift(1)
    df['r_close'] = np.log(df['close'] / df['prev_close'])

    # Intraday log returns
    df['r_open'] = np.log(df['open'] / df['close'])
    df['r_high'] = np.log(df['high'] / df['close'])
    df['r_low'] = np.log(df['low'] / df['close'])

    # Volume change
    df['prev_vol'] = grp['volume'].shift(1)
    df['r_vol'] = np.log(df['volume'] / df['prev_vol'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward fill, then drop remaining NaNs
    df = (df.groupby('symbol')
          .apply(lambda x: x.ffill())
          .reset_index(drop=True))
    df = df.dropna(subset=['r_close', 'r_open', 'r_high', 'r_low', 'r_vol'])

    if debug:
        nan_ct = df[['r_close', 'r_open', 'r_high', 'r_low', 'r_vol']].isna().sum()
        print("[DEBUG] NaN count:\n", nan_ct.to_dict())
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2) Macro Data Loading
# ════════════════════════════════════════════════════════════════════════════
def get_macro_data(macro: str,
                   *,
                   region: str = "USA",
                   use_index: bool = False,
                   start: str = START_DATE,
                   end: str = TEST_END) -> pd.DataFrame:
    # Load Macro CSV from data directory
    macro_file = get_data_file_path(f"macro_{macro.lower()}.csv")
    df = pd.read_csv(macro_file, on_bad_lines="skip")

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', case=False)]
    df = df.loc[:, df.columns != '']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Load Index CSV if requested
    if use_index:
        idx_file = get_data_file_path(f"index_{_REGION_TAG[region.upper()]}.csv")

        # Read only Date and Price
        idx = pd.read_csv(idx_file,
                          usecols=[0, 1],
                          names=["date", "index"],
                          header=0)

        idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
        df = pd.merge(df, idx, on="date", how="left")

        if use_index and "index" not in RAW_MACRO and "index" in df.columns:
            RAW_MACRO.append("index")
            if "index" not in NO_LOGRET:
                NO_LOGRET.append("index")

    # Align dates and forward fill
    price_dates = pd.date_range(start, end, freq='B')

    df = df.set_index('date') \
        .reindex(price_dates) \
        .sort_index() \
        .ffill()

    df.index.name = 'date'
    df = df.reset_index()

    # Shift macro columns for non-USA regions (aligning time zones)
    if region.upper() not in {"USA", "CRYPTO"}:
        cols_to_shift = [c for c in RAW_MACRO if c != "index"]
        df[cols_to_shift] = df[cols_to_shift].shift(1).ffill()

    # Scaling and Log Returns
    for col in RAW_MACRO:
        roll_min = df[col].expanding().min()
        roll_max = df[col].expanding().max()
        df[f"{col}_scaled"] = (df[col] - roll_min) / (roll_max - roll_min + 1e-9)

    for col in (c for c in RAW_MACRO if c not in NO_LOGRET):
        df[f"{col}_logret"] = np.log(df[col] / df[col].shift(1))

    global MACRO_COLS
    MACRO_COLS = RAW_MACRO + [f"{c}_scaled" for c in RAW_MACRO] + \
                 [f"{c}_logret" for c in RAW_MACRO if c not in NO_LOGRET]

    return df[['date'] + MACRO_COLS].ffill().dropna()


# ════════════════════════════════════════════════════════════════════════════
# 3) Tensor Construction
# ════════════════════════════════════════════════════════════════════════════
def _key_dates(pr: pd.DataFrame):
    base = pr.groupby('symbol').size().idxmax()
    return pr.loc[pr['symbol'] == base, 'date'].sort_values().unique()


def build_temporal_graph(pr_df: pd.DataFrame,
                         macro_df: pd.DataFrame,
                         lags: int = 5,
                         progress_sym: str = "AAPL",
                         debug=False) -> Dict[str, torch.Tensor]:
    """
    Constructs the graph tensor.
    Includes an external 'Others' node to represent cash/outside market flow.
    """
    symbols = pr_df['symbol'].unique().tolist()
    if len(symbols) == 0:
        raise ValueError("No symbols remaining in the dataset.")

    if progress_sym not in symbols:
        if debug:
            print(f"[WARN] Reference symbol '{progress_sym}' not found. Using '{symbols[0]}'.")
        progress_sym = symbols[0]

    n, full_dates = len(symbols), _key_dates(pr_df)
    dates = full_dates[lags:]
    T = len(dates)

    # 1) Node Features (Price/Volume)
    price_cols = ['r_close', 'r_open', 'r_high', 'r_low', 'r_vol']
    node_feat = (pr_df[pr_df['date'].isin(dates)]
                 .pivot(index='date', columns='symbol', values=price_cols)
                 .reindex(dates).fillna(0.0)
                 .values.reshape(T, n, len(price_cols)))
    node_feat = torch.tensor(node_feat, dtype=torch.float32)

    # Sector & Macro Features
    sector_tbl = pr_df.groupby('symbol')['sector'].first().reindex(symbols).tolist()
    sec2idx = {s: i for i, s in enumerate(sorted(set(sector_tbl)))}
    sec_onehot = torch.zeros(n, len(sec2idx))
    for i, s in enumerate(sector_tbl): sec_onehot[i, sec2idx[s]] = 1
    sector_feat = sec_onehot.repeat(T, 1, 1)

    macro_feat = torch.tensor(macro_df.set_index('date')
                              .loc[dates, MACRO_COLS].values,
                              dtype=torch.float32).unsqueeze(1).repeat(1, n, 1)

    # 2) Market Cap Matrix
    cap_raw = (pr_df[pr_df['date'].isin(dates)]
               .pivot(index='date', columns='symbol', values='market_cap')
               .reindex(dates)[symbols]
               .ffill()
               .fillna(0.0)
               )

    # 3) Calculate Flow (F) and 'Others' node
    F_full, cap_norm_rows, cap_raw_rows = [], [], []
    for t in range(1, T):
        cap_t = cap_raw.iloc[t - 1].values.astype(float)
        cap_t1 = cap_raw.iloc[t].values.astype(float)

        cap_t = np.nan_to_num(cap_t, nan=0.0)
        cap_t1 = np.nan_to_num(cap_t1, nan=0.0)

        total_t = cap_t.sum()
        if total_t == 0:
            F_block = np.zeros((n + 1, n + 1), dtype=np.float32)
            F_full.append(F_block)
            cap_norm_rows.append(np.zeros(n + 1))
            cap_raw_rows.append(np.zeros(n + 1))
            continue

        c_norm_t = cap_t / total_t
        g = (cap_t1 - cap_t) / total_t
        g_ext = -g.sum()

        g_all = np.concatenate([g, [g_ext]])
        m = n + 1
        lam = g_all / m
        F = lam[None, :] - lam[:, None]
        np.fill_diagonal(F, 0.0)

        row_scale = np.concatenate([c_norm_t, [1.0]])
        row_scale[row_scale == 0] = 1e-9
        F[:n] = F[:n] / row_scale[:n, None]

        F_block = np.zeros((n + 1, n + 1), dtype=np.float32)
        F_block[:] = F
        F_full.append(F_block)

        cap_norm_rows.append(np.concatenate([c_norm_t, [0.0]]))
        cap_raw_rows.append(np.concatenate([cap_t, [abs(g_ext) * total_t]]))

    if not F_full:
        raise ValueError("F_full is empty. Insufficient dates.")

    zero_mat = np.zeros_like(F_full[0])
    F_full = torch.tensor([zero_mat] + F_full, dtype=torch.float32)

    # 4) Pad features for the 'Others' node and create Lag Stack
    pad_zeros = torch.zeros(T, 1, node_feat.size(-1))
    node_feat = torch.cat([node_feat, pad_zeros], dim=1)

    sec_pad = torch.zeros(T, 1, sector_feat.size(-1))
    sector_feat = torch.cat([sector_feat, sec_pad], dim=1)

    macro_pad = torch.zeros(T, 1, macro_feat.size(-1))
    macro_feat = torch.cat([macro_feat, macro_pad], dim=1)

    F_lag = torch.stack(
        [torch.roll(F_full, shifts=i, dims=0) for i in range(1, lags + 1)],
        dim=1)[:T - lags]

    X_static = torch.cat([node_feat, sector_feat, macro_feat], dim=-1)[lags:]

    data = {
        "X_static": X_static,
        "F_lag": F_lag,
        "F_label": F_full[lags:],
        "cap_norm": torch.tensor(cap_norm_rows, dtype=torch.float32)[lags - 1:],
        "cap_raw": torch.tensor(cap_raw_rows, dtype=torch.float32)[lags - 1:],
        "dates": dates[lags:],
        "symbols": symbols + ["Others"]
    }
    return data


def preprocess_pipeline(*,
                        region: str,
                        macro: str = "USA",
                        use_index: bool = False,
                        debug: bool = False,
                        lags: int = 5,
                        cutoff_date: Optional[str] = None) -> None:
    """
    Executes the full preprocessing pipeline and saves the result to cache.
    """
    # 1) Load Price Data
    pr = preprocess_data(get_prices(region), debug=debug)

    # 2) Apply Cutoff Filter (remove symbols appearing after training start)
    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        first_dates = pr.groupby('symbol')['date'].min()
        print(f"First dates summary:\n{first_dates.head()}")

        keep_syms = first_dates[first_dates <= cutoff_ts].index.tolist()

        if not keep_syms:
            raise ValueError(
                f"[FILTER-ERROR] No symbols found starting before {cutoff_date}. "
                "Check the train_from date or dataset coverage."
            )

        dropped = sorted(set(first_dates.index) - set(keep_syms))
        pr = pr[pr['symbol'].isin(keep_syms)]
        if debug:
            print(f"[FILTER] cutoff={cutoff_date} kept={len(keep_syms)} dropped={len(dropped)}")

    # 3) Load Macro Data
    mc = get_macro_data(macro, region=region, use_index=use_index)

    # 4) Build Tensor
    data = build_temporal_graph(pr, mc, lags=lags, debug=debug)

    # 5) Save
    save_dataset(lags=lags,
                 cutoff_date=cutoff_date or "FULL",
                 region=region,
                 macro=macro,
                 use_index=use_index,
                 **data)