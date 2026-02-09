###############################################################################
# main.py  ·  preprocess & DDP training entry
###############################################################################
import argparse
import pickle
import torch
import torch.multiprocessing as mp
import os
import random
import numpy as np
import pandas as pd
import socket
import pathlib
import traceback
import sys
from torch.utils.data import TensorDataset, DataLoader
from engine import build_model
import engine
import preprocess
from utils_path import dataset_path, _REGION_TAG


# -----------------------------------------------------------------------------
# Configuration & Path Utils
# -----------------------------------------------------------------------------
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _best_ckpt_path(args):
    # Determine target folder based on Special Periods (MASTER/MGDPR)
    target_folder = args.region

    # USA Special: 2020-01-01 ~ 2022-12-31
    if args.region == "USA" and args.train_from == "2010-01-01" and \
            args.test_from == "2020-01-01" and args.test_to == "2022-12-31":
        target_folder = "MASTER"

    # CHN Special: 2018-01-01 ~ 2019-12-31
    elif args.region == "CHN" and args.train_from == "2010-01-01" and \
            args.test_from == "2018-01-01" and args.test_to == "2019-12-31":
        target_folder = "MGDPR"

    dir_path = pathlib.Path.cwd() / "checkpoints" / target_folder
    dir_path.mkdir(parents=True, exist_ok=True)

    fname = (f"best_{args.model}_lag{args.lags}_{args.train_from}_{args.valid_from}_{args.test_from}"
             f"_ep{args.epochs}_bs{args.batch}_seed{args.seed}_k{args.gcn_k}_{args.beta}"
             f"_m{_REGION_TAG[args.macro.upper()]}"
             f"{'_index' if args.index else ''}"
             f"_h{args.d_hidden}_l{args.gcn_layers}_tf{args.t_filters}_tk{args.t_kernel}.pt")

    return str(dir_path / fname)


def _cached_data_path(args):
    return dataset_path(args.lags, args.train_from,
                        args.region, args.macro,
                        use_index=args.index)


# -----------------------------------------------------------------------------
# DDP Setup
# -----------------------------------------------------------------------------
def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _setup_ddp_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_dataset(args):
    path = _cached_data_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run preprocess first: python main.py --preprocess ..."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["X_static"], data["F_lag"], data["F_label"], data["dates"]


def build_loader(Xs, Fl, F, idx, bs=16, shuffle=False):
    ds = TensorDataset(Xs[idx], Fl[idx], F[idx])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# -----------------------------------------------------------------------------
# Training / Validation
# -----------------------------------------------------------------------------
def _l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    return sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, opt, device,
                    *, use_hp: bool, beta: float) -> float:
    model.train()
    loss_sum = 0.0

    for step, (Xs, Fl, F_true) in enumerate(loader):
        Xs, Fl, F_true = (Xs.to(device), Fl.to(device), F_true.to(device))

        F_pred = model(Xs, Fl)
        F_out = engine.maybe_project(F_pred, F_true.sum(-1), use_hp=use_hp)

        mse = torch.nn.functional.mse_loss(F_out, F_true)
        reg = beta * _l2_penalty(model) if beta > 0 else 0.0
        loss = mse + reg

        # Error Check
        if not torch.isfinite(loss):
            raise ValueError(f"Loss is {loss.item()} (NaN/Inf) at step {step}")

        opt.zero_grad()
        loss.backward()

        # Gradient Check
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError(f"Gradient NaN/Inf detected at {name} (step {step})")

        opt.step()
        loss_sum += loss.item()

    return loss_sum / len(loader)


@torch.no_grad()
def valid_loop(model, loader, device, *, use_hp: bool) -> float:
    model.eval()
    loss_sum = 0.0
    for Xs, Fl, F_true in loader:
        Xs, Fl, F_true = (Xs.to(device), Fl.to(device), F_true.to(device))
        F_pred = model(Xs, Fl)
        F_out = engine.maybe_project(F_pred, F_true.sum(-1), use_hp=use_hp)
        loss_sum += torch.nn.functional.mse_loss(F_out, F_true).item()
    return loss_sum / len(loader)


# -----------------------------------------------------------------------------
# DDP Worker
# -----------------------------------------------------------------------------
def ddp_worker(rank, world_size, args):
    try:
        _set_seed(args.seed + rank)
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

        Xs, Fl, F, dates = load_dataset(args)

        # Date splitting
        str_dates = [str(d)[:10] for d in dates]
        idx_tr = [i for i, d in enumerate(str_dates) if args.train_from <= d < args.valid_from]
        idx_vl = [i for i, d in enumerate(str_dates) if args.valid_from <= d < args.test_from]

        if len(idx_tr) == 0:
            raise ValueError(
                f"Training set empty! Check --train_from ({args.train_from}) and --valid_from ({args.valid_from})")

        tr_loader = build_loader(Xs, Fl, F, idx_tr, args.batch, True)
        vl_loader = build_loader(Xs, Fl, F, idx_vl, args.batch, False) if idx_vl else None

        # Model Build
        model = build_model(
            args.model, Xs.size(-1), Xs.size(1),
            d_hidden=args.d_hidden, tau=args.lags,
            gcn_k=args.gcn_k, gcn_norm=args.gcn_norm,
            gcn_layers=args.gcn_layers, t_filters=args.t_filters, t_kernel=args.t_kernel
        ).cuda()

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_loss = 1e9

        for ep in range(1, args.epochs + 1):
            if args.model == "emb_stgcn":
                freeze = ep < args.w_thres
                if hasattr(model.module, "flow_mem"):
                    for p in model.module.flow_mem.parameters():
                        p.requires_grad_(not freeze)

            tr_loss = train_one_epoch(model, tr_loader, opt, f'cuda:{rank}',
                                      use_hp=args.hard_projection, beta=args.beta)

            if vl_loader:
                val_loss = valid_loop(model, vl_loader, f'cuda:{rank}', use_hp=args.hard_projection)
            else:
                val_loss = tr_loss

            if rank == 0:
                if ep % 10 == 0 or ep == 1:
                    # [Modified] Minimal log to indicate progress without showing MSE values
                    print(f"[Epoch {ep}/{args.epochs}] Training...", flush=True)

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.module.state_dict(), _best_ckpt_path(args))

        if rank == 0:
            print(f"[Main] Training Finished.")

    except Exception:
        # Traceback on error
        print(f"!!! Error in DDP Worker {rank} !!!", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--region", required=True)
    a.add_argument("--macro", default="USA")
    a.add_argument("--index", action="store_true")
    a.add_argument("--train_from", required=True)
    a.add_argument("--valid_from", default="2024-01-01")
    a.add_argument("--test_from", default="2025-01-01")
    a.add_argument("--test_to", default="2025-03-31")

    a.add_argument("--model", default="emb_stgcn")
    a.add_argument("--lags", type=int, default=20)
    a.add_argument("--epochs", type=int, default=200)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--beta", type=float, default=0.0)
    a.add_argument("--seed", type=int, default=123)

    a.add_argument("--d_hidden", type=int, default=64)
    a.add_argument("--gcn_layers", type=int, default=2)
    a.add_argument("--t_filters", type=int, default=16)
    a.add_argument("--t_kernel", type=int, default=5)
    a.add_argument("--gcn_k", type=int, default=2)
    a.add_argument("--gcn_norm", default="deg")
    a.add_argument("--w_thres", type=int, default=10)

    a.add_argument("--preprocess", action='store_true')
    a.add_argument("--k", type=int, default=3)
    a.add_argument("--fee_bp", type=float, default=0.0)
    a.add_argument("--hard_projection", default=True)
    a.add_argument("--verbose", action='store_true')
    a.add_argument("--debug", action='store_true')

    # Unused but kept for compatibility
    a.add_argument("--monitor_epochs", type=str, default="")
    a.add_argument("--plot_date", default=None)
    a.add_argument("--compare_date", default=None)
    a.add_argument("--edge_pct", type=float, default=0.1)
    a.add_argument("--analyze_others", action='store_true')
    a.add_argument("--simulate", action='store_true')

    args = a.parse_args()

    # Handle str input for boolean if passed from shell
    if isinstance(args.hard_projection, str):
        args.hard_projection = args.hard_projection.lower() == 'true'

    _set_seed(args.seed)

    if args.preprocess:
        preprocess.preprocess_pipeline(region=args.region, macro=args.macro,
                                       use_index=args.index, debug=args.debug,
                                       lags=args.lags, cutoff_date=args.train_from)
        return

    # Check Checkpoint
    ckpt = _best_ckpt_path(args)
    if not os.path.exists(ckpt):
        print(f"[Info] Training started... (Target: {ckpt})")
        _setup_ddp_env()
        mp.spawn(ddp_worker, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(), args))
    else:
        print(f"[Info] Checkpoint exists. Skipping training.")


if __name__ == "__main__":
    main()