"""Single-image inference latency benchmark for DualBranchRoadNet.

Companion to compare_reparameterization.py, which measures Params/GMACs/Peak
memory but not latency. This script reuses its checkpoint-loading and
deploy-conversion logic so all three numbers in Table~\\ref{tab:efficiency}
come from a consistent model instance.

Protocol: FP32, eval mode, single-image (B=1) forward at 1024x1024, no TTA --
matching the "FLOPs path" already described in the paper
(one B=1 forward, no TTA), so GMACs and latency are measured under the same
input shape and batch size.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from compare_reparameterization import load_training_model, main_logits


@torch.inference_mode()
def measure_latency(
    model_cpu: nn.Module,
    height: int,
    width: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    model = model_cpu.to(device).eval()
    x = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)

    for _ in range(warmup):
        main_logits(model(x))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        main_logits(model(x))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    model.to("cpu")
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "iters": iters,
        "warmup": warmup,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_img_per_s": float(1000.0 / arr.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Single-image FP32 latency benchmark for DualBranchRoadNet",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--form",
        choices=("deploy", "multi", "both"),
        default="deploy",
        help="Use deploy for the final model row in a resource-comparison table",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.height, args.width, args.warmup, args.iters) < 1:
        raise ValueError("height, width, warmup and iters must be positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for the latency benchmark")

    training_model, checkpoint, checkpoint_path = load_training_model(args.ckpt, args.weights)
    training_model.eval()

    deploy_model = copy.deepcopy(training_model)
    if not hasattr(deploy_model, "switch_to_deploy"):
        raise AttributeError("Model has no switch_to_deploy() method")
    deploy_model.switch_to_deploy()
    deploy_model.eval()

    print("=" * 88)
    print("LATENCY BENCHMARK (single-image, FP32, no TTA)")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Weights    : {args.weights} | epoch={checkpoint.get('epoch')}")
    print(f"GPU        : {torch.cuda.get_device_name(device)}")
    print(f"Input      : 1x3x{args.height}x{args.width} | FP32 | eval mode")
    print(f"Warmup     : {args.warmup} iters | Timed: {args.iters} iters")

    selected: List[Tuple[str, nn.Module]] = []
    if args.form in {"multi", "both"}:
        selected.append(("MULTI-BRANCH FORM", training_model))
    if args.form in {"deploy", "both"}:
        selected.append(("REPARAMETERIZED DEPLOY FORM", deploy_model))

    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "gpu": torch.cuda.get_device_name(device),
        "input": [1, 3, args.height, args.width],
        "forms": {},
    }

    for label, model in selected:
        stats = measure_latency(model, args.height, args.width, device, args.warmup, args.iters)
        result["forms"][label] = stats
        print("-" * 88)
        print(label)
        print(
            f"Latency      : {stats['mean_ms']:.3f} ms mean | "
            f"median={stats['median_ms']:.3f} | std={stats['std_ms']:.3f} | "
            f"min={stats['min_ms']:.3f} | max={stats['max_ms']:.3f}"
        )
        print(f"Throughput   : {stats['throughput_img_per_s']:.2f} img/s")

    print("=" * 88)
    print("For Table 5, use DEPLOY latency (mean, ms) at B=1, 1024x1024, FP32.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON : {out}")


if __name__ == "__main__":
    main()
