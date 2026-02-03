from __future__ import annotations
import torch
from .boxes import clamp_boxes01


def _intersection_box(bA: torch.Tensor, bB: torch.Tensor) -> torch.Tensor:
    # bA,bB: (4,)
    x1 = torch.maximum(bA[0], bB[0])
    y1 = torch.maximum(bA[1], bB[1])
    x2 = torch.minimum(bA[2], bB[2])
    y2 = torch.minimum(bA[3], bB[3])
    return torch.stack([x1, y1, x2, y2])


def _bridge_box(bA: torch.Tensor, bB: torch.Tensor) -> torch.Tensor:
    # minimal rectangle spanning both boxes (works when fully separated or as fallback)
    x1 = torch.minimum(bA[0], bB[0])
    y1 = torch.minimum(bA[1], bB[1])
    x2 = torch.maximum(bA[2], bB[2])
    y2 = torch.maximum(bA[3], bB[3])
    return torch.stack([x1, y1, x2, y2])


def _bridge_strip(bA: torch.Tensor, bB: torch.Tensor, axis: str) -> torch.Tensor:
    """
    axis='x' means boxes overlap in x but not in y -> vertical strip between them.
    axis='y' means boxes overlap in y but not in x -> horizontal strip between them.
    """
    if axis == "x":
        # x overlap segment
        x1 = torch.maximum(bA[0], bB[0])
        x2 = torch.minimum(bA[2], bB[2])
        # y gap between boxes
        top = torch.minimum(bA[3], bB[3])
        bottom = torch.maximum(bA[1], bB[1])
        # If A above B: y1=top_of_lower, y2=bottom_of_upper
        y1 = torch.minimum(top, bottom)
        y2 = torch.maximum(top, bottom)
        return torch.stack([x1, y1, x2, y2])

    if axis == "y":
        # y overlap segment
        y1 = torch.maximum(bA[1], bB[1])
        y2 = torch.minimum(bA[3], bB[3])
        # x gap between boxes
        left = torch.minimum(bA[2], bB[2])
        right = torch.maximum(bA[0], bB[0])
        x1 = torch.minimum(left, right)
        x2 = torch.maximum(left, right)
        return torch.stack([x1, y1, x2, y2])

    raise ValueError("axis must be 'x' or 'y'")


def compute_relation_box(bA01: torch.Tensor, bB01: torch.Tensor) -> torch.Tensor:
    """
    Compute relation box br from two normalized boxes bA and bB (both in [0,1]).

    Args:
        bA01: (4,) [x1,y1,x2,y2]
        bB01: (4,) [x1,y1,x2,y2]

    Returns:
        br01: (4,) normalized relation box
    """
    if bA01.shape != (4,) or bB01.shape != (4,):
        raise ValueError(f"bA01 and bB01 must be shape (4,), got {bA01.shape} and {bB01.shape}")

    bA = clamp_boxes01(bA01.view(1, 4))[0]
    bB = clamp_boxes01(bB01.view(1, 4))[0]

    # Ensure ordering x1<=x2, y1<=y2
    bA = torch.tensor([min(bA[0], bA[2]), min(bA[1], bA[3]), max(bA[0], bA[2]), max(bA[1], bA[3])],
                      device=bA.device, dtype=bA.dtype)
    bB = torch.tensor([min(bB[0], bB[2]), min(bB[1], bB[3]), max(bB[0], bB[2]), max(bB[1], bB[3])],
                      device=bB.device, dtype=bB.dtype)

    x_ol = torch.minimum(bA[2], bB[2]) - torch.maximum(bA[0], bB[0])
    y_ol = torch.minimum(bA[3], bB[3]) - torch.maximum(bA[1], bB[1])

    overlap_x = (x_ol >= 0)
    overlap_y = (y_ol >= 0)

    if overlap_x and overlap_y:
        br = _intersection_box(bA, bB)
    elif overlap_x and (not overlap_y):
        br = _bridge_strip(bA, bB, axis="x")
    elif (not overlap_x) and overlap_y:
        br = _bridge_strip(bA, bB, axis="y")
    else:
        br = _bridge_box(bA, bB)

    # Clamp and ensure ordering (in case strip becomes degenerate)
    br = br.clamp(0.0, 1.0)
    x1, y1, x2, y2 = br
    br = torch.stack([torch.minimum(x1, x2), torch.minimum(y1, y2),
                      torch.maximum(x1, x2), torch.maximum(y1, y2)])
    return br


def compute_relation_boxes_from_pairs(boxes01: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """
    Compute relation boxes for many pairs.

    Args:
        boxes01: (N,4)
        pairs: (M,2) long tensor, each row [i,j]

    Returns:
        rel_boxes01: (M,4)
    """
    if boxes01.ndim != 2 or boxes01.shape[1] != 4:
        raise ValueError(f"boxes01 must be (N,4), got {tuple(boxes01.shape)}")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must be (M,2), got {tuple(pairs.shape)}")

    rel_boxes = []
    for k in range(pairs.shape[0]):
        i = int(pairs[k, 0].item())
        j = int(pairs[k, 1].item())
        rel_boxes.append(compute_relation_box(boxes01[i], boxes01[j]))
    return torch.stack(rel_boxes, dim=0) if rel_boxes else torch.zeros((0, 4), dtype=boxes01.dtype, device=boxes01.device)
