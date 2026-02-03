from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch


def _draw_rect(ax, box01, color="k", linewidth=2, label=None):
    x1, y1, x2, y2 = [float(v) for v in box01]
    w = x2 - x1
    h = y2 - y1
    rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)
    if label is not None:
        ax.text(x1, y1, label, color=color, fontsize=10, verticalalignment="bottom")


def plot_boxes_and_masks(
    boxes01: torch.Tensor,
    rel_box01: torch.Tensor,
    obj_masks: torch.Tensor,
    rel_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    title: str = ""
):
    """
    Visualize:
    - normalized boxes A/B and relation box
    - object masks, relation mask, background mask (H x W)
    """
    boxes_np = boxes01.detach().cpu().numpy()
    rel_np = rel_box01.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title)

    # Panel 1: boxes in normalized coordinates
    ax0 = plt.subplot(1, 4, 1)
    ax0.set_title("Boxes (norm)")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.invert_yaxis()  # visual like image coords (top=0)
    ax0.set_aspect("equal")

    # draw objects
    colors = ["g", "b", "c", "m", "y"]
    for i in range(min(len(boxes_np), len(colors))):
        _draw_rect(ax0, boxes_np[i], color=colors[i], linewidth=2, label=f"obj{i}")
    # draw relation
    _draw_rect(ax0, rel_np, color="k", linewidth=2, label="rel")

    # Panel 2: object union
    ax1 = plt.subplot(1, 4, 2)
    ax1.set_title("Object union mask")
    union = obj_masks.detach().cpu().numpy().max(axis=0)
    ax1.imshow(union, interpolation="nearest")
    ax1.axis("off")

    # Panel 3: relation mask
    ax2 = plt.subplot(1, 4, 3)
    ax2.set_title("Relation mask")
    ax2.imshow(rel_mask.detach().cpu().numpy(), interpolation="nearest")
    ax2.axis("off")

    # Panel 4: background mask
    ax3 = plt.subplot(1, 4, 4)
    ax3.set_title("Background mask")
    ax3.imshow(bg_mask.detach().cpu().numpy(), interpolation="nearest")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()
