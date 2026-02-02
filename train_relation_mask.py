from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.toy_relation_mask_dataset import ToyRelationMaskDataset, ToyRelMaskConfig
from src.models.relation_mask_predictor import (
    BaselineRelMaskPredictor,
    PredicateConditionedRelMaskPredictor,
    SubjObjPredicateConditionedRelMaskPredictor,
    OracleConditionedRelMaskPredictor,
)



def batch_iou(pred_prob: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_prob: (B,1,H,W) in [0,1]
    gt:        (B,1,H,W) in {0,1}
    """
    pred = (pred_prob > thr).float()
    inter = (pred * gt).sum(dim=(1,2,3))
    union = ((pred + gt) > 0).float().sum(dim=(1,2,3))
    return (inter + eps) / (union + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "pred", "pred2", "oracle"], default="baseline")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_n", type=int, default=5000)
    parser.add_argument("--val_n", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ToyRelMaskConfig(H=64, W=64, n_objects=2, seed=0)
    train_ds = ToyRelationMaskDataset(args.train_n, cfg)
    val_ds = ToyRelationMaskDataset(args.val_n, cfg)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    if args.mode == "baseline":
       model = BaselineRelMaskPredictor(n_objects=2)
    elif args.mode == "pred":
       model = PredicateConditionedRelMaskPredictor(n_objects=2, n_predicates=5, pred_dim=8)
    elif args.mode == "pred2":
       model = SubjObjPredicateConditionedRelMaskPredictor(n_predicates=5, pred_dim=8, rel_dim=16)
    else:
       model = OracleConditionedRelMaskPredictor(n_objects=2)


    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            obj_masks = batch["obj_masks"].to(args.device)      # (B,2,H,W)
            bg_mask = batch["bg_mask"].to(args.device)          # (B,H,W)
            rel_mask = batch["rel_mask"].to(args.device)        # (B,H,W)

            gt = rel_mask.unsqueeze(1)                          # (B,1,H,W)
            pred_id = batch["pred_id"].to(args.device).long()  # (B,)

            if args.mode == "baseline":
               logits = model(obj_masks, bg_mask)
            elif args.mode in ["pred", "pred2"]:
               logits = model(obj_masks, bg_mask, pred_id)
            else:
               logits = model(obj_masks, bg_mask, rel_mask)


            loss = F.binary_cross_entropy_with_logits(logits, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                obj_masks = batch["obj_masks"].to(args.device)
                bg_mask = batch["bg_mask"].to(args.device)
                rel_mask = batch["rel_mask"].to(args.device)
                gt = rel_mask.unsqueeze(1)

                pred_id = batch["pred_id"].to(args.device).long()  # (B,)

                if args.mode == "baseline":
                   logits = model(obj_masks, bg_mask)
                elif args.mode in ["pred", "pred2"]:
                   logits = model(obj_masks, bg_mask, pred_id)
                else:  # oracle
                   logits = model(obj_masks, bg_mask, rel_mask)


                loss = F.binary_cross_entropy_with_logits(logits, gt)
                val_loss += loss.item()

                prob = torch.sigmoid(logits)
                ious.append(batch_iou(prob, gt).cpu())

        mean_iou = torch.cat(ious).mean().item()
        print(
            f"[{args.mode}] epoch {epoch:02d} "
            f"train_loss={running_loss/len(train_loader):.4f} "
            f"val_loss={val_loss/len(val_loader):.4f} "
            f"val_IoU={mean_iou:.4f}"
        )


if __name__ == "__main__":
    main()
