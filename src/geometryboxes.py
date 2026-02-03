from __future__ import annotations
import torch


def clamp_boxes01(boxes: torch.Tensor) -> torch.Tensor:
    """
    Clamp normalized boxes into [0,1].

    Args:
        boxes: Tensor (N,4) in normalized coords [0,1]

    Returns:
        Tensor (N,4) clamped to [0,1]
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N,4), got {tuple(boxes.shape)}")
    return boxes.clamp(0.0, 1.0)


def box_to_mask(boxes01: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert normalized bounding boxes to binary masks.

    Args:
        boxes01: Tensor (N,4), each box = [x1,y1,x2,y2] in [0,1]
        H, W: mask size (recommend 64x64 for SD latent)

    Returns:
        masks: Tensor (N,H,W) with values in {0,1} (dtype=torch.float32)
    """
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive integers")

    boxes01 = clamp_boxes01(boxes01).to(dtype=torch.float32)
    N = boxes01.shape[0]
    if N == 0:
        return torch.zeros((0, H, W), dtype=torch.float32, device=boxes01.device)

    # Convert normalized coords to pixel indices
    # Use half-open interval [x1, x2) and [y1, y2) for stable masking.
    x1 = (boxes01[:, 0] * W).floor().to(torch.long)
    y1 = (boxes01[:, 1] * H).floor().to(torch.long)
    x2 = (boxes01[:, 2] * W).ceil().to(torch.long)
    y2 = (boxes01[:, 3] * H).ceil().to(torch.long)

    # Clamp indices
    x1 = x1.clamp(0, W)
    x2 = x2.clamp(0, W)
    y1 = y1.clamp(0, H)
    y2 = y2.clamp(0, H)

    # Ensure well-ordered corners
    x1_, x2_ = torch.minimum(x1, x2), torch.maximum(x1, x2)
    y1_, y2_ = torch.minimum(y1, y2), torch.maximum(y1, y2)

    # Build grid
    xs = torch.arange(W, device=boxes01.device).view(1, 1, W)   # (1,1,W)
    ys = torch.arange(H, device=boxes01.device).view(1, H, 1)   # (1,H,1)

    # Broadcast comparisons -> (N,H,W)
    in_x = (xs >= x1_.view(N, 1, 1)) & (xs < x2_.view(N, 1, 1))
    in_y = (ys >= y1_.view(N, 1, 1)) & (ys < y2_.view(N, 1, 1))
    masks = (in_x & in_y).to(torch.float32)
    return masks


def union_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Union over N masks.

    Args:
        masks: (N,H,W)

    Returns:
        (H,W) union mask
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N,H,W), got {tuple(masks.shape)}")
    if masks.shape[0] == 0:
        return torch.zeros(masks.shape[1:], dtype=torch.float32, device=masks.device)
    return masks.to(torch.float32).amax(dim=0)


def background_mask_from_object_masks(obj_masks: torch.Tensor) -> torch.Tensor:
    """
    Background mask = 1 outside all object masks.

    Args:
        obj_masks: (N,H,W) in {0,1}

    Returns:
        (H,W) background mask
    """
    u = union_masks(obj_masks)
    return (1.0 - u).clamp(0.0, 1.0)
