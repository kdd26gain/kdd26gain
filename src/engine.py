###############################################################################
# engine.py  ·  Models & Metrics
###############################################################################
import torch.nn as nn
import math, matplotlib.pyplot as plt, networkx as nx, numpy as np, torch
import itertools
from stgcn import STGCN, EmbSTGCN


# ────────────────── Hard-Projection ─────────────────────────────────────────
def hard_project(F_raw: torch.Tensor, delta_cap: torch.Tensor) -> torch.Tensor:
    B = 0.5 * (F_raw - F_raw.transpose(-1, -2))
    n = F_raw.size(-1)
    lam = (delta_cap / n).unsqueeze(-1)  # (B,n,1)
    F_star = B + (lam - lam.transpose(-1, -2))
    return F_star - torch.diag_embed(torch.diagonal(F_star, dim1=-2, dim2=-1))


# ────────────────── Helper for Baseline ────────────────────────────────────
def flatten_lag_edges(F_lag: torch.Tensor) -> torch.Tensor:
    B, tau, n, _ = F_lag.shape
    return F_lag.permute(0, 2, 1, 3).reshape(B, n, tau * n)


# ──────────────────  Attention Baseline (Old GAIN)  ────────────────────────
class GAIN(nn.Module):
    def __init__(self, d_static: int, n_nodes: int,
                 tau: int,
                 d_hidden: int = 64):
        super().__init__()
        self.n = n_nodes
        self.tau = tau

        edge_dim = tau * n_nodes
        self.proj_static = nn.Linear(d_static + edge_dim, d_hidden)
        self.proj_Frow = nn.Linear(n_nodes, d_hidden, bias=False)
        self.attn = nn.MultiheadAttention(d_hidden, 4, batch_first=True)
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_hidden) * 0.02)

    def forward(self, X_static: torch.Tensor, F_lag: torch.Tensor, verbose=False):
        B, lags, n, _ = F_lag.shape
        F_edge = flatten_lag_edges(F_lag)
        X_cat = torch.cat([X_static, F_edge], -1)

        Xs = self.proj_static(X_cat)
        rows = F_lag.transpose(2, 3).reshape(B * n, lags, n)
        rows = self.proj_Frow(rows)

        attn_out, _ = self.attn(rows, rows, rows)
        h = attn_out[:, -1]
        U = h.reshape(B, n, -1) + Xs

        S = torch.matmul(U, self.W_dec)
        score = torch.matmul(S, U.transpose(1, 2))
        F_raw = score - score.transpose(1, 2)
        return F_raw


# ────────────────── Hard-Projection util ──────────────────
def maybe_project(F_raw: torch.Tensor,
                  delta_cap: torch.Tensor,
                  use_hp: bool) -> torch.Tensor:
    if use_hp:
        return hard_project(F_raw, delta_cap)
    return F_raw


def draw_F_graph(F, symbols, caps_raw,
                 edge_top_pct=1.0, title=None, save_path=None, figsize=(6, 6), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if isinstance(F, torch.Tensor): F = F.cpu().numpy()
    if isinstance(caps_raw, torch.Tensor): caps_raw = caps_raw.cpu().numpy()

    n = F.shape[0]
    G = nx.DiGraph()
    for i, lab in enumerate(symbols): G.add_node(i, label=lab)

    for i in range(n):
        for j in range(n):
            if F[i, j] > 0: G.add_edge(i, j, weight=F[i, j])

    all_abs = [abs(w) for *_, w in G.edges(data="weight")]
    if not all_abs:
        ax.set_title(title or "")
        ax.axis("off")
        return
    q = np.quantile(all_abs, 1.0 - edge_top_pct)
    G.remove_edges_from([(u, v) for u, v, w in G.edges(data="weight") if abs(w) < q])

    caps_norm = caps_raw / caps_raw.max()
    node_sizes = 300 + 4000 * caps_norm
    row_sum = F.sum(axis=1)
    max_abs = max(abs(row_sum).max(), 1e-9)

    def _color(val):
        if abs(val) < 0.01 * max_abs: return (0.6, 0.6, 0.6)
        if val > 0: return (0, 0.2, 0.4 + 0.6 * (val / max_abs))
        return (0.4 + 0.6 * (-val / max_abs), 0, 0)

    node_colors = [_color(v) for v in row_sum]
    pos = nx.spring_layout(G, k=3.0 / math.sqrt(n), seed=42, iterations=600)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: s for i, s in enumerate(symbols)}, font_size=8, ax=ax)

    edge_w = [w for *_, w in G.edges(data="weight")]
    if edge_w:
        nx.draw_networkx_edges(G, pos, edge_color=edge_w, edge_cmap=plt.cm.Reds,
                               arrows=True, width=[2.5 * w / max(edge_w) for w in edge_w], ax=ax)
    ax.set_title(title or "")
    ax.axis("off")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")


def draw_F_pair(F_pred, F_true, symbols, caps_raw,
                edge_top_pct=0.1, date_str="", save_path=None, figsize=(12, 6)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    draw_F_graph(F_pred, symbols, caps_raw, edge_top_pct, f"Predicted {date_str}", ax=axs[0])
    draw_F_graph(F_true, symbols, caps_raw, edge_top_pct, f"Ground-truth {date_str}", ax=axs[1])
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")


def compute_others_share(F_series: torch.Tensor, symbols: list):
    if "Others" not in symbols: raise ValueError("No Others node")
    idx = symbols.index("Others")
    abs_F = F_series.abs() if isinstance(F_series, torch.Tensor) else torch.from_numpy(np.abs(F_series))
    total = abs_F.sum(dim=(-2, -1))
    others = abs_F[:, idx, :].sum(-1) + abs_F[:, :, idx].sum(-1)
    ratios = others / (total + 1e-12)
    return torch.nanmean(ratios).item(), ratios


_MODEL_ZOO = {"attention": GAIN, "stgcn": STGCN, "emb_stgcn": EmbSTGCN}


def build_model(model_name: str, d_static: int, n_nodes: int, **kwargs):
    if model_name not in _MODEL_ZOO:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name == "attention":
        tau = kwargs.pop("tau")
        return GAIN(d_static, n_nodes, tau=tau, **kwargs)
    return _MODEL_ZOO[model_name](d_static, n_nodes, **kwargs)


@torch.no_grad()
def quick_simulation_metrics(model, Xs, Fl, ret, *, k=5, fee_mul=0.9995):
    model.eval()
    n = ret.shape[1]
    nav, r_vec = [1.0], []
    for t in range(len(ret)):
        F_raw = model(Xs[t:t + 1], Fl[t:t + 1]).squeeze(0)
        F_hat = maybe_project(F_raw, F_raw.sum(-1), use_hp=True).cpu().numpy()

        flow = F_hat[:-1].sum(1)
        long_idx = flow.argsort()[:k]
        short_idx = flow.argsort()[-k:]
        abs_w = np.r_[np.abs(flow[long_idx]), np.abs(flow[short_idx])]

        w = np.zeros(n)
        w[long_idx] = abs_w[:k] / abs_w.sum()
        w[short_idx] = -abs_w[k:] / abs_w.sum()

        r_day = (w * ret[t]).sum()
        r_day = (1 + r_day) * (fee_mul ** np.abs(w).sum()) - 1
        r_vec.append(r_day)
        nav.append(nav[-1] * (1 + r_day))

    nav_arr = np.asarray(nav)
    r_arr = np.asarray(r_vec)
    cumr = (nav_arr[-1] - 1) * 100
    sharpe = 0.0 if r_arr.std() == 0 else (r_arr.mean() / r_arr.std()) * np.sqrt(252)
    mdd = (np.maximum.accumulate(nav_arr) - nav_arr).max() / nav_arr.max()
    return cumr, sharpe, mdd