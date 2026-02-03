from __future__ import annotations
import torch
import torch.nn as nn


class SmallConvMaskNet(nn.Module):
    """
    Input:  (B,C,H,W)
    Output: (B,1,H,W) logits
    """
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineRelMaskPredictor(nn.Module):
    """
    Uses only obj_masks + bg_mask as input.
    Channels: [obj0, obj1, bg] => C=3
    """
    def __init__(self, n_objects: int = 2):
        super().__init__()
        self.n_objects = n_objects
        self.core = SmallConvMaskNet(in_ch=n_objects + 1)

    def forward(self, obj_masks: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
        # obj_masks: (B,N,H,W), bg_mask: (B,H,W)
        x = torch.cat([obj_masks, bg_mask.unsqueeze(1)], dim=1)  # (B,N+1,H,W)
        return self.core(x)  # (B,1,H,W)


class OracleConditionedRelMaskPredictor(nn.Module):
    """
    Upper bound: Uses GT rel_mask as extra conditioning channel.
    Channels: [obj0, obj1, bg, rel_gt] => C=4
    """
    def __init__(self, n_objects: int = 2):
        super().__init__()
        self.n_objects = n_objects
        self.core = SmallConvMaskNet(in_ch=n_objects + 2)

    def forward(self, obj_masks: torch.Tensor, bg_mask: torch.Tensor, rel_mask_cond: torch.Tensor) -> torch.Tensor:
        # rel_mask_cond: (B,H,W)
        x = torch.cat([obj_masks, bg_mask.unsqueeze(1), rel_mask_cond.unsqueeze(1)], dim=1)
        return self.core(x)

class PredicateConditionedRelMaskPredictor(nn.Module):
    """
    Condition on predicate via an embedding broadcasted to (H,W).
    Channels: [obj0, obj1, bg] + [pred_emb channels]
    """
    def __init__(self, n_objects: int = 2, n_predicates: int = 5, pred_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.n_objects = n_objects
        self.pred_dim = pred_dim
        self.pred_emb = nn.Embedding(n_predicates, pred_dim)
        self.core = SmallConvMaskNet(in_ch=n_objects + 1 + pred_dim, hidden=hidden)

    def forward(self, obj_masks: torch.Tensor, bg_mask: torch.Tensor, pred_id: torch.Tensor) -> torch.Tensor:
        """
        obj_masks: (B,N,H,W)
        bg_mask:  (B,H,W)
        pred_id:  (B,) long
        """
        B, _, H, W = obj_masks.shape
        e = self.pred_emb(pred_id)                 # (B,pred_dim)
        e_map = e.view(B, self.pred_dim, 1, 1).expand(B, self.pred_dim, H, W)  # (B,pred_dim,H,W)

        x = torch.cat([obj_masks, bg_mask.unsqueeze(1), e_map], dim=1)
        return self.core(x)  # (B,1,H,W)
    
import torch
import torch.nn as nn


class SubjObjPredicateConditionedRelMaskPredictor(nn.Module):
    """
    Condition on predicate + (subj,obj) geometric features extracted from obj_masks.

    Inputs:
      obj_masks: (B,2,H,W) in {0,1}
      bg_mask:   (B,H,W)
      pred_id:   (B,) long

    Output:
      logits:    (B,1,H,W)
    """
    def __init__(self, n_predicates: int = 5, pred_dim: int = 8, rel_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.pred_emb = nn.Embedding(n_predicates, pred_dim)

        # MLP that fuses: [subj_geom(3), obj_geom(3), pred_emb(pred_dim)] -> rel_feat(rel_dim)
        in_dim = 3 + 3 + pred_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, rel_dim),
            nn.ReLU(inplace=True),
        )

        # final conv net takes: [obj0, obj1, bg] + rel_feat_map(rel_dim)  => 3+rel_dim channels
        self.core = SmallConvMaskNet(in_ch=3 + rel_dim, hidden=hidden)

    @staticmethod
    def _geom_from_mask(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        m: (B,H,W) mask in {0,1}
        return: (B,3) = [area, cx, cy] where cx,cy in [0,1]
        """
        B, H, W = m.shape
        device = m.device

        # coordinate grids normalized to [0,1]
        ys = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1).expand(B, H, W)
        xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, W).expand(B, H, W)

        mass = m.sum(dim=(1, 2)) + eps          # (B,)
        area = mass / float(H * W)              # normalized area in [0,1] roughly

        cx = (m * xs).sum(dim=(1, 2)) / mass    # (B,)
        cy = (m * ys).sum(dim=(1, 2)) / mass    # (B,)

        return torch.stack([area, cx, cy], dim=1)  # (B,3)

    def forward(self, obj_masks: torch.Tensor, bg_mask: torch.Tensor, pred_id: torch.Tensor) -> torch.Tensor:
        # obj_masks: (B,2,H,W)
        B, N, H, W = obj_masks.shape
        assert N == 2, "This toy model assumes exactly 2 objects."

        pred_id = pred_id.long()
        e = self.pred_emb(pred_id)  # (B,pred_dim)

        subj_geom = self._geom_from_mask(obj_masks[:, 0])  # (B,3)
        obj_geom  = self._geom_from_mask(obj_masks[:, 1])  # (B,3)

        rel_feat = self.fuse(torch.cat([subj_geom, obj_geom, e], dim=1))  # (B,rel_dim)
        rel_map = rel_feat.view(B, -1, 1, 1).expand(B, rel_feat.shape[1], H, W)  # (B,rel_dim,H,W)

        x = torch.cat([obj_masks, bg_mask.unsqueeze(1), rel_map], dim=1)  # (B,3+rel_dim,H,W)
        return self.core(x)  # (B,1,H,W)

