###############################################################################
# utils_path.py
# Manages file paths relative to the project structure.
###############################################################################
from pathlib import Path
import os

# Define Project Roots
# Assumes structure:
# project_root/
#   ├── src/ (contains this file)
#   ├── data/
#   └── cache/
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Region tags for file naming conventions
_REGION_TAG = {
    "USA": "usa", "EUR": "eur", "CHN": "chn",
    "KOR": "kor", "SSE": "sse", "FTSE": "ftse"
}


def _ensure_dir(p: Path) -> None:
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def get_data_file_path(filename: str) -> str:
    """Returns the full path for a file in the data directory."""
    return str(DATA_DIR / filename)


def dataset_path(lags: int,
                 cutoff_date: str,
                 region: str,
                 macro: str = "USA",
                 *,
                 use_index: bool = False) -> str:
    """Returns the path for the preprocessed pickle cache."""
    tag = _REGION_TAG[region.upper()]
    mtag = f"m{_REGION_TAG[macro.upper()]}"

    name = f"cache_gain_lag{lags}_{cutoff_date}_{tag}_{mtag}"
    if use_index:
        name += "_index"

    tgt = CACHE_DIR / name
    _ensure_dir(tgt.parent)
    return str(tgt.resolve())


def ckpt_path(train_from: str, valid_from: str, test_from: str,
              lags: int, batch: int, seed: int, model: str,
              region: str, gcn_k: int, beta: float,
              *,
              macro: str = "USA",
              use_index: bool = False) -> str:
    """Returns the path for saving model checkpoints."""
    tag = _REGION_TAG[region.upper()]
    mtag = f"m{_REGION_TAG[macro.upper()]}"

    name = (
        f"best_{model}_lag{lags}_{train_from}_{valid_from}_{test_from}"
        f"_bs{batch}_seed{seed}_k{gcn_k}_{beta}_{tag}_{mtag}"
    )
    if use_index:
        name += "_index"

    # Save checkpoints in a dedicated directory instead of src root
    # or follow specific logic if passed target_folder logic is needed elsewhere.
    tgt = CHECKPOINT_DIR / name
    _ensure_dir(tgt.parent)
    return str(tgt.resolve())


def _best_ckpt_path_generated(args):
    """
    Helper to generate checkpoint path based on arguments,
    including specific folder structures for experimental regimes.
    """
    target_folder = args.region

    # Special handling for specific experimental periods
    if args.region == "USA" and args.train_from == "2010-01-01" and \
            args.test_from == "2020-01-01" and args.test_to == "2022-12-31":
        target_folder = "MASTER"

    elif args.region == "CHN" and args.train_from == "2010-01-01" and \
            args.test_from == "2018-01-01" and args.test_to == "2019-12-31":
        target_folder = "MGDPR"

    dir_path = CHECKPOINT_DIR / target_folder
    dir_path.mkdir(parents=True, exist_ok=True)

    fname = (f"best_{args.model}_lag{args.lags}_{args.train_from}_{args.valid_from}_{args.test_from}"
             f"_ep{args.epochs}_bs{args.batch}_seed{args.seed}_k{args.gcn_k}_{args.beta}"
             f"_m{_REGION_TAG[args.macro.upper()]}"
             f"{'_index' if args.index else ''}"
             f"_h{args.d_hidden}_l{args.gcn_layers}_tf{args.t_filters}_tk{args.t_kernel}.pt")

    return str(dir_path / fname)