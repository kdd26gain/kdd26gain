import os
import glob
import re
import shutil
import subprocess
import pickle
import sys
import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    from engine import build_model, maybe_project
    import preprocess
    from utils_path import dataset_path
except ImportError:
    print("Error: 'engine.py', 'preprocess.py', 'utils_path.py' required.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MODEL_DIR = "model"
OUTPUT_REPORT = "reproduce_report.txt"
MAIN_SCRIPT = "main.py"
PYTHON_EXEC = sys.executable
CACHE_DIR = "cache"

# Fixed start dates for training
TRAIN_START_DATES = {
    "CHN": "2013-01-07",
    "USA": "2013-01-03",
    "EUR": "2013-01-03",
    "KOR": "2018-01-01"
}

# Period Mappings
PERIOD_MAP = {
    1: {"test_from": "2024-10-01", "test_to": "2025-03-31"},
    2: {"test_from": "2024-01-01", "test_to": "2024-09-30"},
    3: {"test_from": "2020-01-01", "test_to": "2020-03-31"},
    100: {"test_from": "2020-01-01", "test_to": "2020-12-31", "special_tag": "THGNN"},
    101: {"test_from": "2015-07-01", "test_to": "2017-12-31", "special_tag": "MGDPR"}
}

DEFAULT_ARGS = {
    "model": "emb_stgcn", "gcn_k": 2, "gcn_norm": "deg",
    "hard_projection": True, "k": 3
}

# -----------------------------------------------------------------------------
# OPTION A: BEST ASR (Only this remains)
# -----------------------------------------------------------------------------
TARGET_SETTINGS_A = {
    # CHN
    ("CHN", 1): {"epoch": 80, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 128, 'gcn_layers': 2, 't_filters': 32,
                            't_kernel': 10, 'lags': 10}},
    ("CHN", 2): {"epoch": 100, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 2,
                            't_filters': 32, 't_kernel': 5, 'lags': 20}},
    ("CHN", 3): {"epoch": 40, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 16, 't_kernel': 5, 'lags': 20}},
    ("CHN", 100): {"epoch": 20, "seeds": range(5),
                   "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 64, 'gcn_layers': 3,
                              't_filters': 32, 't_kernel': 5, 'lags': 20}},
    # EUR
    ("EUR", 1): {"epoch": 120, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 128, 'gcn_layers': 2,
                            't_filters': 16, 't_kernel': 10, 'lags': 10}},
    ("EUR", 2): {"epoch": 80, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 16, 't_kernel': 5, 'lags': 10}},
    ("EUR", 3): {"epoch": 200, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 32, 't_kernel': 5, 'lags': 10}},
    # KOR
    ("KOR", 1): {"epoch": 100, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 16, 't_kernel': 10, 'lags': 10}},
    ("KOR", 2): {"epoch": 120, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 64, 'gcn_layers': 2, 't_filters': 16,
                            't_kernel': 10, 'lags': 20}},
    ("KOR", 3): {"epoch": 120, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 32, 't_kernel': 10, 'lags': 20}},
    # USA
    ("USA", 1): {"epoch": 200, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 128, 'gcn_layers': 3,
                            't_filters': 32, 't_kernel': 10, 'lags': 20}},
    ("USA", 2): {"epoch": 50, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 64, 'gcn_layers': 3,
                            't_filters': 32, 't_kernel': 5, 'lags': 10}},
    ("USA", 3): {"epoch": 40, "seeds": range(5),
                 "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.001, 'd_hidden': 64, 'gcn_layers': 2,
                            't_filters': 16, 't_kernel': 5, 'lags': 20}},
    ("USA", 101): {"epoch": 60, "seeds": range(5),
                   "params": {'beta': 0.0001, 'batch': 32, 'lr': 0.0001, 'd_hidden': 128, 'gcn_layers': 3,
                              't_filters': 32, 't_kernel': 10, 'lags': 10}}
}


# -----------------------------------------------------------------------------
# UTIL FUNCTIONS
# -----------------------------------------------------------------------------

def calculate_dates(p_id):
    p_conf = PERIOD_MAP[p_id]
    v_start = datetime.strptime(p_conf["test_from"], "%Y-%m-%d") - relativedelta(months=3)
    return {
        "valid_from": v_start.strftime("%Y-%m-%d"),
        "test_from": p_conf["test_from"],
        "test_to": p_conf["test_to"]
    }


def get_train_from(region, p_id):
    if region == "CHN" and p_id == 100:
        return "2016-01-01"
    return TRAIN_START_DATES.get(region, "2013-01-01")


def check_cache_exists(region, lags, p_id):
    train_from = get_train_from(region, p_id)
    path = dataset_path(lags, train_from, region, region, use_index=True)
    return os.path.exists(path), path, train_from


def move_newly_created_ckpt(region, start_time_marker, dest_path):
    possible_folders = [
        os.path.join("checkpoints", region),
        os.path.join("checkpoints", "MASTER"),
        os.path.join("checkpoints", "MGDPR")
    ]
    candidates = []
    for folder in possible_folders:
        if not os.path.exists(folder): continue
        for f in glob.glob(os.path.join(folder, "*.pt")):
            mtime = os.path.getmtime(f)
            if mtime > start_time_marker:
                candidates.append((f, mtime))

    if not candidates: return None
    latest_file = max(candidates, key=lambda x: x[1])[0]
    if os.path.exists(dest_path): os.remove(dest_path)
    shutil.move(latest_file, dest_path)
    return dest_path


def calculate_metrics(nav, r_vec):
    r_arr = np.array(r_vec)
    nav_arr = np.array(nav)
    mu = r_arr.mean()
    sigma = r_arr.std() + 1e-9
    asr = (mu / sigma) * np.sqrt(252)
    avol = sigma * np.sqrt(252)
    mdd = 0.0
    peak = nav_arr[0]
    for v in nav_arr:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)

    return {
        "ASR": asr, "RMDD": mdd, "AVol": avol
    }


# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def run_experiment_group(settings_dict, mode_name="ASR"):
    print(f"\n{'=' * 60}\n >>> STARTING {mode_name} MODE (Total configs: {len(settings_dict)})\n{'=' * 60}")

    sorted_keys = sorted(settings_dict.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for (region, p_id) in sorted_keys:
        setting = settings_dict[(region, p_id)]
        params = setting["params"]
        target_epoch = setting["epoch"]
        dates = calculate_dates(p_id)

        real_train_from = get_train_from(region, p_id)
        cache_exists, cache_path, real_train_from = check_cache_exists(region, params['lags'], p_id)

        if not cache_exists:
            print(f"[Skip] Cache not found: {cache_path}")
            continue

        period_tag = f"P{p_id}"
        dest_dir = os.path.join(MODEL_DIR, f"{region}_{period_tag}_{mode_name}")
        os.makedirs(dest_dir, exist_ok=True)

        print(f"\nProcessing {region} {period_tag} ({mode_name}) | Ep: {target_epoch} | TrainFrom: {real_train_from}")

        seed_metrics = []

        for seed in setting["seeds"]:
            final_model_path = os.path.join(dest_dir, f"seed{seed}_ep{target_epoch}.pt")

            # 1. Training (if needed)
            if not os.path.exists(final_model_path):
                print(f"  [Seed {seed}] Training...", end="", flush=True)
                start_time = time.time()
                cmd = [PYTHON_EXEC, MAIN_SCRIPT]
                cmd += ["--region", region, "--macro", region]
                cmd += ["--train_from", real_train_from, "--valid_from", dates['valid_from']]
                cmd += ["--test_from", dates['test_from'], "--test_to", dates['test_to']]
                cmd += ["--seed", str(seed), "--epochs", str(target_epoch), "--index"]
                for k, v in params.items(): cmd += [f"--{k}", str(v)]

                try:
                    result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                            text=True)
                    moved = move_newly_created_ckpt(region, start_time, final_model_path)
                    if not moved:
                        print(" Failed to create model (File missing).")
                        continue
                    print(" Done.")
                except subprocess.CalledProcessError as e:
                    print(" Training Error.")
                    print("-" * 20 + " ERROR LOG " + "-" * 20)
                    print(e.stderr)
                    print("-" * 50)
                    continue
            else:
                print(f"  [Seed {seed}] Found existing model. Skipping training.")

            # 2. Evaluation (Investment metrics only)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            all_dates_pd = pd.to_datetime(data["dates"])
            mask = (all_dates_pd >= pd.to_datetime(dates['test_from'])) & \
                   (all_dates_pd <= pd.to_datetime(dates['test_to']))

            Xs_test = data["X_static"][mask].to(device)
            Fl_test = data["F_lag"][mask].to(device)

            symbols = np.asarray(data["symbols"])[:-1]
            close_test = preprocess.get_prices(region).pivot(
                index='date', columns='symbol', values='close'
            ).reindex(columns=symbols).reindex(all_dates_pd[mask]).ffill()
            ret_test = close_test.pct_change().fillna(0.0).values

            model = build_model(
                DEFAULT_ARGS['model'], Xs_test.size(-1), Xs_test.size(1),
                d_hidden=params['d_hidden'], tau=params['lags'],
                gcn_k=DEFAULT_ARGS['gcn_k'], gcn_norm=DEFAULT_ARGS['gcn_norm'],
                gcn_layers=params['gcn_layers'], t_filters=params['t_filters'],
                t_kernel=params['t_kernel']
            ).to(device)

            sd = torch.load(final_model_path, map_location=device)
            model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
            model.eval()

            nav, r_vec = [1.0], []

            with torch.no_grad():
                for t in range(len(ret_test)):
                    curr_X = Xs_test[t:t + 1]
                    if curr_X.size(-1) < Xs_test.size(-1):
                        curr_X = torch.nn.functional.pad(curr_X, (0, Xs_test.size(-1) - curr_X.size(-1)))
                    F_out = model(curr_X, Fl_test[t:t + 1])
                    F_pred = maybe_project(F_out, F_out.sum(-1), True)

                    F_p_np = F_pred.squeeze(0).cpu().numpy()

                    # Portfolio (Long-Short)
                    row_sum = F_p_np[:-1].sum(1)
                    k_inv = 3
                    w = np.zeros(len(symbols))
                    w[np.argsort(row_sum)[:k_inv]] = 1.0 / k_inv
                    w[np.argsort(row_sum)[-k_inv:]] = -1.0 / k_inv
                    day_ret = np.sum(w * ret_test[t])
                    r_vec.append(day_ret)
                    nav.append(nav[-1] * (1 + day_ret))

            seed_metrics.append(calculate_metrics(nav, r_vec))

        if seed_metrics:
            avg = {}
            for k in ["ASR", "RMDD", "AVol"]:
                avg[k] = np.mean([s[k] for s in seed_metrics])

            log_str = f"[{mode_name}] {region} P{p_id} (Ep {target_epoch}): "
            log_str += (f"ASR={avg['ASR']:.4f}, RMDD={avg['RMDD']:.4f}, AVol={avg['AVol']:.4f}")

            print(f"> {log_str}")
            with open(OUTPUT_REPORT, "a") as f:
                f.write(log_str + "\n")


def main():
    print("Reproduce Pipeline Started...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_REPORT): os.remove(OUTPUT_REPORT)

    # Only Run Best ASR Settings (Target A)
    run_experiment_group(TARGET_SETTINGS_A, mode_name="ASR")

    print(f"\n Reproduction Complete. Results saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()