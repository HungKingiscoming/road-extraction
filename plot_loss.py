"""Plot the training loss curve from a training run.

Reads the metrics.jsonl produced by train.py (one JSON record per epoch,
with a "train" sub-dict) and renders a single-panel PNG of the total training
loss per epoch.

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


def extract_train_curve(records: List[Dict]):
    epochs, losses = [], []
    for record in records:
        train = record.get("train") or {}
        if "total" in train:
            epochs.append(record["epoch"])
            losses.append(train["total"])
    return epochs, losses


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

    epochs, losses = extract_train_curve(load_records(log_path))
    if not epochs:
        raise RuntimeError(f"No train loss records found in {log_path}")

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(epochs, losses, label="train loss", color="tab:blue")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Training loss")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    print(f"Final train loss: {losses[-1]:.5f} at epoch {epochs[-1]}")


if __name__ == "__main__":
    main()
