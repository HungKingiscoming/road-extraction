"""Plot train vs. validation loss (and calibrated IoU) from a training run.

Reads the metrics.jsonl produced by train.py (one JSON record per epoch,
with "train" and "validation" sub-dicts) and renders a two-panel PNG:
top panel is train loss vs. val_loss_main (CE+Dice on the native-resolution
validation set), bottom panel is calibrated road IoU for context.

val_loss_main intentionally excludes the centerline auxiliary term that
train loss includes -- RoadReconstructionDecoder only computes that head in
training mode, and forcing eval-mode BatchNorm/dropout into a training-mode
forward pass just to get it would corrupt the validation signal far more
than the missing (aux_weight=0.15-scaled) term costs. Watch the *shape* of
the two curves (does val_loss_main flatten/rise while train loss keeps
falling) rather than their absolute gap, since the formulas differ by one
smaller-weighted term.

Usage:
    python plot_loss.py --log checkpoints/roadfusion_scratch/metrics.jsonl
    python plot_loss.py --log metrics.jsonl --out curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: Kaggle/servers have no display to render to
import matplotlib.pyplot as plt


def load_records(log_path: Path) -> List[Dict]:
    records = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {log_path}")
    return records


def extract_curves(records: List[Dict]):
    train_epochs, train_losses = [], []
    val_epochs, val_losses, val_ious = [], [], []
    for record in records:
        epoch = record["epoch"]
        train = record.get("train") or {}
        if "total" in train:
            train_epochs.append(epoch)
            train_losses.append(train["total"])
        validation = record.get("validation") or {}
        if "val_loss_main" in validation:
            val_epochs.append(epoch)
            val_losses.append(validation["val_loss_main"])
            val_ious.append(validation.get("calibrated_road_iou"))
    return train_epochs, train_losses, val_epochs, val_losses, val_ious


def summarize_overfitting(
    val_epochs: List[int], val_losses: List[float], val_ious: List[float]
) -> str:
    if not val_epochs:
        return "No validation records with val_loss_main found."
    best_loss_index = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_iou_index = max(
        range(len(val_ious)), key=lambda i: (val_ious[i] is not None, val_ious[i] or -1)
    )
    lines = [
        f"Best val loss:  {val_losses[best_loss_index]:.5f} at epoch {val_epochs[best_loss_index]}",
        f"Best cal. IoU:  {val_ious[best_iou_index]:.5f} at epoch {val_epochs[best_iou_index]}"
        if val_ious[best_iou_index] is not None
        else "Best cal. IoU:  n/a",
        f"Final val loss: {val_losses[-1]:.5f} at epoch {val_epochs[-1]}",
    ]
    if val_epochs[best_loss_index] < val_epochs[-1] - max(5, len(val_epochs) // 10):
        lines.append(
            "-> Best val loss happened well before the final epoch; the "
            "later epochs may be overfitting (or just riding a flat LR near "
            "the end of the cosine schedule -- check whether calibrated IoU "
            "also peaked early)."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to metrics.jsonl")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <log dir>/loss_curves.png)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_name("loss_curves.png")

    records = load_records(log_path)
    train_epochs, train_losses, val_epochs, val_losses, val_ious = extract_curves(records)

    figure, (loss_axis, iou_axis) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    loss_axis.plot(train_epochs, train_losses, label="train loss", color="tab:blue")
    loss_axis.plot(val_epochs, val_losses, label="val loss (CE+Dice)", color="tab:orange")
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Train vs. validation loss")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    iou_epochs = [e for e, v in zip(val_epochs, val_ious) if v is not None]
    iou_values = [v for v in val_ious if v is not None]
    iou_axis.plot(iou_epochs, iou_values, label="calibrated road IoU", color="tab:green")
    iou_axis.set_xlabel("epoch")
    iou_axis.set_ylabel("IoU")
    iou_axis.set_title("Calibrated road IoU (for overfitting context)")
    iou_axis.legend()
    iou_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    print()
    print(summarize_overfitting(val_epochs, val_losses, val_ious))


if __name__ == "__main__":
    main()
