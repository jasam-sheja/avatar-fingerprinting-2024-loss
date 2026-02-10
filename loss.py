import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn


def _pdist2(x: Tensor) -> Tensor:
    """Compute pair-wise squared Euclidean distances."""
    x_norm = x.pow(2).sum(-1, keepdim=True)
    sq_dist = x_norm.add(x_norm.transpose(-2, -1)).baddbmm_(
        x, x.transpose(-2, -1), beta=1, alpha=-2
    )
    return torch.relu_(sq_dist)


def _cdist2(x1: Tensor, x2: Tensor) -> Tensor:
    """Compute pair-wise squared Euclidean distances."""
    x1_norm = x1.pow(2).sum(-1, keepdim=True)
    x2_norm = x2.pow(2).sum(-1, keepdim=True)
    sq_dist = torch.baddbmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), beta=1, alpha=-2
    ).add_(x1_norm)
    return torch.relu_(sq_dist)


class DynamicIdentityLoss(nn.Module):
    """
    Implements equations (1)-(5) from
    'Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos'.
    You must supply:
        emb          - (B, D, T)   embeddings for all clips in the batch
        shemb        - (B, D, T)   embeddings for shuffled clips in the batch
        driver_id    - (B,)     long tensor of driver identity indices
        source_id    - (B,)     long tensor of source identity indices
        driver_id_sh - (B,) long tensor of shuffled driver identity indices
    """

    def __init__(
        self,
        eps: float = 1e-12,
        temperature: float = 0.5,
        apply_shuffle: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.eps = eps
        self.tau = temperature
        self.apply_shuffle = apply_shuffle
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be one of 'mean', 'sum', or 'none', but got {reduction}"
            )
        self.reduction = reduction

    def extra_repr(self):
        return (
            f"temperature={self.tau}, "
            f"apply_shuffle={self.apply_shuffle}, reduction={self.reduction}"
            f"eps={self.eps}"
        )

    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(
        self,
        emb: Tensor,
        driver_id: Tensor,
        source_id: Tensor,
        sh_emb: Tensor = None,
        driver_id_sh: Tensor = None,
    ):
        apply_shuffle = (
            self.apply_shuffle and sh_emb is not None and driver_id_sh is not None
        )
        if emb.dtype not in (torch.float32, torch.float64):
            # ensure float precision for distance calculations, but allow half precision for storage
            emb = emb.float()
        if apply_shuffle and sh_emb.dtype not in (torch.float32, torch.float64):
            # ensure float precision for distance calculations, but allow half precision for storage
            sh_emb = sh_emb.float()

        B, D, T = emb.shape
        emb = rearrange(emb, "B D T -> (B T) D")

        # ---------- pair-wise similarities in log-space ----------
        dist = _pdist2(emb[None])[0]  # (B*T, B*T)
        scale = (
            dist.mean() + self.eps
        )  # stabilize scale; this is not in the paper but helps convergence
        sim = dist / (-scale * self.tau + self.eps)  # no exp, stays finite

        sim = reduce(
            sim,
            "(B1 T1) (B2 T2) -> B1 T1 B2",
            "max",
            B1=B,
            T1=T,
            B2=B,
            T2=T,
        )

        # ---------- masks ----------
        same_drv = driver_id.unsqueeze(1) == driver_id.unsqueeze(0)
        same_src = source_id.unsqueeze(1) == source_id.unsqueeze(0)

        nm = -float("inf")  # log(0) for masking

        # Pull (same driver)
        pull = sim.masked_fill(~same_drv[:, None, :], nm)
        logN = torch.logsumexp(pull, dim=-1)  # (B,T) → (B*T)

        # Push (same source, different driver)
        push = sim.masked_fill(~(same_src & ~same_drv)[:, None, :], nm)
        logQ = torch.logsumexp(push, dim=-1)

        if apply_shuffle:
            if sh_emb.dtype not in (torch.float32, torch.float64):
                # ensure float precision for distance calculations, but allow half precision for storage
                sh_emb = sh_emb.float()
            sh = rearrange(sh_emb, "B D T -> (B T) D")
            # ---------- pair-wise similarities in log-space ----------
            dist_sh = _cdist2(emb[None], sh[None])[0]
            sim_sh = dist_sh / (-scale * self.tau + self.eps)
            sim_sh = reduce(
                sim_sh,
                "(B T) (Bsh Tsh) -> B T Bsh",
                "max",
                B=B,
                T=T,
                Bsh=B,
                Tsh=T,
            )
            # ---------- masks ----------
            same_drv_sh = driver_id.unsqueeze(1) == driver_id_sh.unsqueeze(0)
            # Shuffle negatives
            shuf = sim_sh.masked_fill(~same_drv_sh[:, None, :], nm)
            logR = torch.logsumexp(shuf, dim=-1)
            # ---------- log-softmax over [N, Q, R] ----------
            logits = torch.stack((logN, logQ, logR), dim=0)  # (3, B*T)
        else:
            logits = torch.stack((logN, logQ), dim=0)  # (2, B*T)
        logp = F.log_softmax(logits, dim=0)[0]  # p = N / (N+Q+R)

        loss = -logp

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


if __name__ == "__main__":
    # Example usage
    T = 5
    verts = torch.randn(
        4, 5, 68, 3
    )  # 4 samples, 10 time steps, 68 vertices, 3D coordinates
    shuffled_indices = torch.randperm(T)
    # Apply the shuffled indices to the data
    sh_verts = verts[..., shuffled_indices, :, :]
    # Dummy embeddings
    emb = rearrange(verts, "n t v c -> n (v c) t") / T
    sh_emb = rearrange(sh_verts, "n t v c -> n (v c) t") / T
    driver_id = torch.tensor([0, 1, 0, 1])  # driver identities
    source_id = torch.tensor([0, 0, 1, 1])  # source identities

    loss_fn = DynamicIdentityLoss()
    loss = loss_fn(emb, driver_id, source_id, sh_emb, driver_id)
    print("Loss:", loss)
    loss_no_shuffle = loss_fn(emb, driver_id, source_id)
    print("Loss without shuffle:", loss_no_shuffle)
