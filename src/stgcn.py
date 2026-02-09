# stgcn.py
import torch
import torch.nn as nn
import math


###############################################################################
#  FlowMemory : Section 3.3 Firm-Specific Flow Memory
###############################################################################
class FlowMemory(nn.Module):
    def __init__(self, n_nodes: int, d_mem: int = 64, d_out: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.register_buffer("mem", torch.zeros(n_nodes, d_mem))

        # Input dimension must match total nodes (N+1) -> vector A_{i,:}
        self.gru = nn.GRUCell(input_size=n_nodes, hidden_size=d_mem)
        self.proj = nn.Linear(d_mem, d_out)

    def forward(self, F_t: torch.Tensor) -> torch.Tensor:
        """
        F_t : (B, n, n) ─ Latest flow matrix (Row i is outflow from i)
        Returns: (B, n, d_emb)
        """
        out = []
        for b in range(F_t.size(0)):
            flow_vec = F_t[b]

            # GRU Update: input=(n, n_nodes), hidden=(n, d_mem)
            new_mem = self.gru(flow_vec, self.mem)

            # Projection
            E_b = self.proj(new_mem)
            out.append(E_b)

            # Memory Update (In-place)
            self.mem.data.copy_(new_mem.detach())

        return torch.stack(out, 0)


# ───────────────── util : Deg-Norm ──────────────────
def deg_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    deg = A.sum(-1)
    D_inv_sqrt = (deg + eps).rsqrt()
    return D_inv_sqrt.unsqueeze(-1) * A * D_inv_sqrt.unsqueeze(-2)


# ───────────────── Temporal 1×τ Conv ────────────────
class TemporalConv(nn.Module):
    def __init__(self, d_in: int, d_out: int, tau: int, kernel_size: int = None):
        super().__init__()
        if kernel_size is None:
            k_size = (1, tau)
        else:
            k_size = (1, kernel_size)

        self.conv = nn.Conv2d(d_in, d_out, kernel_size=k_size, bias=False)
        self.d_out = d_out

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : (B, τ, n, d_in)
        """
        # Convert to (B, d_in, n, τ)
        x = x_seq.permute(0, 3, 2, 1)
        h = self.conv(x)

        # Take the last time step (sequence end)
        h = h[..., -1]  # (B, d_out, n)

        return h.permute(0, 2, 1)  # (B, n, d_out)


# ───────────────── Spatial K-hop GCN ────────────────
class SpatialGCN(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int = 2, norm: str = "deg"):
        super().__init__()
        self.k, self.norm = k, norm
        self.W = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor, A_raw: torch.Tensor) -> torch.Tensor:
        A = A_raw + torch.eye(A_raw.size(-1), device=A_raw.device)
        if self.norm == "deg":
            A = deg_normalize(A)

        h, out = x, 0.0
        for _ in range(self.k):
            h = torch.matmul(A, h)
            out = out + h

        return self.W(out)


# ───────────────────── ST-GCN ───────────────────────
class STGCN(nn.Module):
    def __init__(self, d_static: int, n_nodes: int,
                 d_hidden: int = 64, tau: int = 5,
                 gcn_k: int = 2, gcn_norm: str = "deg",
                 gcn_layers: int = 2,
                 t_filters: int = 16,
                 t_kernel: int = 5):

        super().__init__()
        self.tau, self.n = tau, n_nodes

        # 1) Static Feature Projection
        self.proj_static = nn.Linear(d_static, d_hidden)

        # 2) Temporal Branch
        self.temporal = TemporalConv(1, t_filters, tau, kernel_size=t_kernel)
        self.tau_proj = nn.Linear(t_filters, d_hidden)

        # 3) Spatial GCN Layers (Stacking)
        self.gcn_layers = nn.ModuleList()
        for _ in range(gcn_layers):
            self.gcn_layers.append(
                SpatialGCN(d_hidden, d_hidden, k=gcn_k, norm=gcn_norm)
            )

        # 4) Decoder
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_hidden) * 0.02)

    def forward(self, X_static: torch.Tensor, F_lag: torch.Tensor, verbose: bool = False):
        B, tau, n, _ = F_lag.shape

        # (A) Static Branch
        H0 = self.proj_static(X_static)

        # (B) Temporal Branch
        row_abs_seq = F_lag.abs().sum(-1).unsqueeze(-1)
        H_tau = self.temporal(row_abs_seq)
        H_tau = self.tau_proj(H_tau)

        # Combine
        H = H0 + H_tau

        # (C) Spatial GCN Stack
        A_raw = F_lag[:, -1].abs()
        for layer in self.gcn_layers:
            H = layer(H, A_raw)

        G = H

        # (D) Bilinear Decoder
        S = torch.matmul(G, self.W_dec)
        score = torch.matmul(S, G.transpose(1, 2))
        F_raw = score - score.transpose(1, 2)

        return F_raw


class EmbSTGCN(STGCN):
    """
    Proposed Method (GAIN) Implementation
    X = [Static_Feat, Flow_Memory_Emb]
    """

    def __init__(self, d_static: int, n_nodes: int,
                 d_hidden: int = 64, tau: int = 5,
                 gcn_k: int = 2, gcn_norm: str = "deg",
                 d_emb: int = 64,
                 gcn_layers: int = 2,
                 t_filters: int = 16,
                 t_kernel: int = 5):
        super().__init__(d_static + d_emb, n_nodes,
                         d_hidden, tau, gcn_k, gcn_norm,
                         gcn_layers=gcn_layers,
                         t_filters=t_filters,
                         t_kernel=t_kernel)

        self.flow_mem = FlowMemory(n_nodes, d_mem=d_emb, d_out=d_emb)

    def forward(self, X_static, F_lag,
                verbose: bool = False,
                require_emb_grad: bool = True):
        # Flow Memory calculation
        with torch.set_grad_enabled(require_emb_grad):
            E_temp = self.flow_mem(F_lag[:, -1])

        if not require_emb_grad:
            E_temp = E_temp.detach()

        # Glocal Feature Combine
        X_aug = torch.cat([X_static, E_temp], -1)

        return super().forward(X_aug, F_lag, verbose)