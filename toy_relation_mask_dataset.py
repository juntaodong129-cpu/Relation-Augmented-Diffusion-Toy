from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset

from src.geometry.boxes import box_to_mask, background_mask_from_object_masks
from src.geometry.relation_box import compute_relation_box


@dataclass
class ToyRelMaskConfig:
    H: int = 64
    W: int = 64
    n_objects: int = 2          # 先固定2个，避免collate复杂
    min_size: float = 0.15
    max_size: float = 0.55
    seed: int = 0


def _rand_uniform(g: torch.Generator, low: float, high: float) -> float:
    
    return (low + (high - low) * torch.rand(1, generator=g).item())

def _rand_box01(g: torch.Generator, min_size: float, max_size: float) -> torch.Tensor:
    w = _rand_uniform(g, min_size, max_size)
    h = _rand_uniform(g, min_size, max_size)
    x1 = _rand_uniform(g, 0.0, 1.0 - w)
    y1 = _rand_uniform(g, 0.0, 1.0 - h)
    x2 = x1 + w
    y2 = y1 + h
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

def _box_iou01(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    # a,b: (4,) in [0,1] with (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + eps)


def compute_pred_id(box_subj: torch.Tensor, box_obj: torch.Tensor, overlap_thr: float = 0.05) -> int:
    """
    5-class predicate:
      0: left_of
      1: right_of
      2: above
      3: below
      4: overlap
    Rule: if IoU > overlap_thr => overlap else compare center deltas.
    """
    iou = _box_iou01(box_subj, box_obj)
    if iou > overlap_thr:
        return 4  # overlap

    sx = (box_subj[0] + box_subj[2]).item() / 2.0
    sy = (box_subj[1] + box_subj[3]).item() / 2.0
    ox = (box_obj[0] + box_obj[2]).item() / 2.0
    oy = (box_obj[1] + box_obj[3]).item() / 2.0

    dx = ox - sx
    dy = oy - sy

    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0   # obj is right/left of subj
    else:
        return 3 if dy > 0 else 2   # obj is below/above subj


class ToyRelationMaskDataset(Dataset):
    """
    Returns:
      - obj_masks: (N,H,W) float in {0,1}
      - bg_mask:   (H,W)   float in {0,1}
      - rel_mask:  (H,W)   float in {0,1}  (GT)
      - boxes01:   (N,4)
      - rel_box01: (4,)
    """
    def __init__(self, n_samples: int, cfg: ToyRelMaskConfig):
        self.n_samples = n_samples
        self.cfg = cfg
        self.g = torch.Generator().manual_seed(cfg.seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use a per-index generator for determinism across epochs
        g = torch.Generator().manual_seed(self.cfg.seed + idx)

        # two random boxes
        boxes01 = torch.stack([
            _rand_box01(g, self.cfg.min_size, self.cfg.max_size),
            _rand_box01(g, self.cfg.min_size, self.cfg.max_size),
        ], dim=0)  # (2,4)

        obj_masks = box_to_mask(boxes01, self.cfg.H, self.cfg.W)  # (2,H,W)
        bg_mask = background_mask_from_object_masks(obj_masks)    # (H,W)
        pred_id = compute_pred_id(boxes01[0], boxes01[1])

        # fixed relation: subj=0, obj=1 (toy)
        rel_box01 = compute_relation_box(boxes01[0], boxes01[1])  # (4,)
        rel_mask = box_to_mask(rel_box01.view(1, 4), self.cfg.H, self.cfg.W)[0]  # (H,W)

        return {
            "obj_masks": obj_masks,           # (2,H,W)
            "bg_mask": bg_mask,               # (H,W)
            "rel_mask": rel_mask,             # (H,W)
            "boxes01": boxes01,               # (2,4)
            "rel_box01": rel_box01,           # (4,)
            "pred_id": torch.tensor(pred_id, dtype=torch.long),
        }
