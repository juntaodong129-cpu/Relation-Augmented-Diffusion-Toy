from __future__ import annotations
import argparse
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.toy_relation_mask_dataset import ToyRelationMaskDataset, ToyRelMaskConfig
from src.models.diffusion_unet import TinyUNet


# ---------- diffusion schedule ----------
def make_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


@torch.no_grad()
def sample_ddpm(model, cond, betas, device, n_steps: int, clip_x0: bool = True):
    """
    DDPM sampling (ancestral).
    Generates x_0 in [-1,1].
    """
    T = betas.shape[0]
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    x = torch.randn(cond.shape[0], 1, cond.shape[2], cond.shape[3], device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, cond, t_batch)

        a_t = alphas[t]
        abar_t = abar[t]
        beta_t = betas[t]

        # predict x0
        x0_pred = (x - torch.sqrt(1 - abar_t) * eps_pred) / torch.sqrt(abar_t)
        if clip_x0:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # posterior mean: μ = 1/sqrt(a_t) * (x_t - beta_t/sqrt(1-abar_t) * eps)
        mean = (x - (beta_t / torch.sqrt(1 - abar_t)) * eps_pred) / torch.sqrt(a_t)

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    return x


# ---------- conditioning builders ----------
def geom_from_mask(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    m: (B,H,W) in {0,1}
    return: (B,3) = [area, cx, cy] in [0,1]
    """
    B, H, W = m.shape
    device = m.device
    ys = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1).expand(B, H, W)
    xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, W).expand(B, H, W)
    mass = m.sum(dim=(1, 2)) + eps
    area = mass / float(H * W)
    cx = (m * xs).sum(dim=(1, 2)) / mass
    cy = (m * ys).sum(dim=(1, 2)) / mass
    return torch.stack([area, cx, cy], dim=1)  # (B,3)


class PredicateEmbedder(torch.nn.Module):
    def __init__(self, n_predicates: int = 5, pred_dim: int = 8):
        super().__init__()
        self.emb = torch.nn.Embedding(n_predicates, pred_dim)

    def forward(self, pred_id: torch.Tensor, H: int, W: int) -> torch.Tensor:
        e = self.emb(pred_id.long())  # (B,pred_dim)
        return e.view(e.shape[0], e.shape[1], 1, 1).expand(e.shape[0], e.shape[1], H, W)


def build_cond(mode: str, obj_masks: torch.Tensor, bg_mask: torch.Tensor, pred_id: torch.Tensor, pred_embedder: PredicateEmbedder | None):
    """
    Returns cond map (B,C,H,W)
    """
    B, N, H, W = obj_masks.shape
    base = torch.cat([obj_masks, bg_mask.unsqueeze(1)], dim=1)  # (B,3,H,W) since N=2

    if mode == "baseline":
        return base

    if mode == "pred":
        assert pred_embedder is not None
        pmap = pred_embedder(pred_id, H, W)  # (B,pred_dim,H,W)
        return torch.cat([base, pmap], dim=1)

    if mode == "pred2":
        # add simple geometric summaries as extra channels (broadcast)
        subj_geom = geom_from_mask(obj_masks[:, 0])  # (B,3)
        obj_geom = geom_from_mask(obj_masks[:, 1])   # (B,3)
        g = torch.cat([subj_geom, obj_geom], dim=1)  # (B,6)
        gmap = g.view(B, 6, 1, 1).expand(B, 6, H, W) # (B,6,H,W)

        if pred_embedder is not None:
            pmap = pred_embedder(pred_id, H, W)
            return torch.cat([base, gmap, pmap], dim=1)
        return torch.cat([base, gmap], dim=1)

    raise ValueError(f"Unknown cond mode: {mode}")


def save_grid(tensor01: torch.Tensor, path: str, nrow: int = 8):
    """
    tensor01: (B,1,H,W) in [0,1]
    Save a simple grid using matplotlib (no seaborn).
    """
    import matplotlib.pyplot as plt

    B, _, H, W = tensor01.shape
    ncol = nrow
    nrow_grid = math.ceil(B / ncol)

    fig = plt.figure(figsize=(ncol, nrow_grid))
    for i in range(B):
        ax = fig.add_subplot(nrow_grid, ncol, i + 1)
        ax.imshow(tensor01[i, 0].cpu().numpy(), vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=["baseline", "pred", "pred2"], default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--train_n", type=int, default=5000)
    parser.add_argument("--val_n", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pred_dim", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="runs_diffmask")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg = ToyRelMaskConfig(H=64, W=64, n_objects=2, seed=0)
    train_ds = ToyRelationMaskDataset(args.train_n, cfg)
    val_ds = ToyRelationMaskDataset(args.val_n, cfg)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    betas = make_linear_beta_schedule(args.T).to(args.device)

    pred_embedder = None
    if args.cond in ["pred", "pred2"]:
        pred_embedder = PredicateEmbedder(n_predicates=5, pred_dim=args.pred_dim).to(args.device)

    # determine cond channels
    # base=3, pred adds pred_dim, pred2 adds 6 (+ pred_dim optional)
    cond_ch = 3
    if args.cond == "pred":
        cond_ch = 3 + args.pred_dim
    elif args.cond == "pred2":
        cond_ch = 3 + 6 + args.pred_dim

    model = TinyUNet(in_ch=1, cond_ch=cond_ch, base_ch=64, time_dim=128).to(args.device)

    params = list(model.parameters()) + ([] if pred_embedder is None else list(pred_embedder.parameters()))
    opt = torch.optim.Adam(params, lr=args.lr)

    # precompute alphas
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if pred_embedder is not None:
            pred_embedder.train()

        for batch in train_loader:
            obj_masks = batch["obj_masks"].to(args.device)        # (B,2,H,W)
            bg_mask = batch["bg_mask"].to(args.device)            # (B,H,W)
            rel_mask = batch["rel_mask"].to(args.device)          # (B,H,W)
            pred_id = batch["pred_id"].to(args.device)            # (B,)

            # x0 in [-1,1]
            x0 = rel_mask.unsqueeze(1) * 2.0 - 1.0                # (B,1,H,W)

            # sample t and noise
            B = x0.shape[0]
            t = torch.randint(0, args.T, (B,), device=args.device, dtype=torch.long)
            eps = torch.randn_like(x0)

            abar_t = abar[t].view(B, 1, 1, 1)
            x_t = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * eps

            cond = build_cond(args.cond, obj_masks, bg_mask, pred_id, pred_embedder)

            eps_pred = model(x_t, cond, t)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1

        # quick validation: sample a batch and compute IoU vs GT after thresholding
        model.eval()
        if pred_embedder is not None:
            pred_embedder.eval()

        with torch.no_grad():
            batch = next(iter(val_loader))
            obj_masks = batch["obj_masks"].to(args.device)
            bg_mask = batch["bg_mask"].to(args.device)
            rel_mask = batch["rel_mask"].to(args.device)
            pred_id = batch["pred_id"].to(args.device)

            cond = build_cond(args.cond, obj_masks, bg_mask, pred_id, pred_embedder)
            x_gen = sample_ddpm(model, cond, betas, args.device, n_steps=args.T, clip_x0=True)  # (B,1,H,W) in [-1,1]

            # to [0,1]
            gen01 = (x_gen + 1.0) / 2.0
            gt01 = rel_mask.unsqueeze(1)

            # hard IoU
            pred_bin = (gen01 > 0.5).float()
            gt_bin = (gt01 > 0.5).float()
            inter = (pred_bin * gt_bin).sum(dim=(1,2,3))
            union = ((pred_bin + gt_bin) > 0).float().sum(dim=(1,2,3))
            iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()

            print(f"[diff-{args.cond}] epoch {epoch:02d} mse_train~{loss.item():.4f} val_gen_IoU={iou:.4f}")

            # save a grid every epoch
            save_grid(gen01[:64], os.path.join(args.save_dir, f"gen_{args.cond}_e{epoch:02d}.png"), nrow=8)
            save_grid(gt01[:64],  os.path.join(args.save_dir, f"gt_e{epoch:02d}.png"), nrow=8)


if __name__ == "__main__":
    main()
