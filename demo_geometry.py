from __future__ import annotations
import torch

from src.datasets.toy_dataset import get_toy_samples
from src.geometry.boxes import box_to_mask, background_mask_from_object_masks
from src.geometry.relation_box import compute_relation_box
from src.vis.plot_boxes import plot_boxes_and_masks


def main():
    H = W = 64  # latent-size masks
    samples = get_toy_samples()

    for s in samples:
        objs = s["objects"]
        rel = s["relations"][0]
        subj = rel["subj"]
        obj = rel["obj"]

        boxes01 = torch.tensor([o["box"] for o in objs], dtype=torch.float32)  # (N,4)
        obj_masks = box_to_mask(boxes01, H, W)  # (N,H,W)

        # relation box from subject and object boxes
        br01 = compute_relation_box(boxes01[subj], boxes01[obj])  # (4,)
        rel_mask = box_to_mask(br01.view(1, 4), H, W)[0]  # (H,W)

        bg_mask = background_mask_from_object_masks(obj_masks)  # (H,W)

        title = f"{s['name']} | prompt: {s['prompt']}"
        plot_boxes_and_masks(
            boxes01=boxes01,
            rel_box01=br01,
            obj_masks=obj_masks,
            rel_mask=rel_mask,
            bg_mask=bg_mask,
            title=title
        )


if __name__ == "__main__":
    main()

from src.models.mask_encoder import MaskEncoder

import torch

encoder = MaskEncoder(in_ch=1, out_ch=16)

mask = torch.zeros(1, 1, 64, 64)   # batch=1, channel=1
feat = encoder(mask)

print(feat.shape)
