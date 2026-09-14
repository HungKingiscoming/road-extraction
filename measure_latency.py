"""Inference latency benchmark for DualBranchRoadNet.

Companion to compare_reparameterization.py, which measures Params/GMACs/Peak
memory but not latency. This script reuses its checkpoint-loading and
deploy-conversion logic so all numbers in Table~\\ref{tab:efficiency} come
from a consistent model instance.

Two protocols are offered:

  single   -- FP32, eval mode, one B=1 forward at 1024x1024, no TTA. Matches
              the GMAC-measurement path in compare_reparameterization.py, but
              is NOT what the released WeavingUnet test code times.

  weaving8 -- Reproduces the actual loop in the released testmassa.py /
              testdg.py: WeavingUnet's `test_one_img_from_path_8` runs FOUR
              sequential forward passes at batch size 2 (8 augmented TTA
              views for one logical image), each followed by a
              .cpu().data.numpy() transfer, then combines the four masks
              with the same flip/rotate numpy ops before the (per-image) wall
              clock in their test scripts advances. That per-image loop is
              what their Table-5 "Inference time" column measures, so a
              direct comparison against DBR-Net requires timing the same
              4x-forward-pass, 8-view pattern rather than a bare B=1 forward.
              Disk I/O (cv2.imread/imwrite) is excluded here since it is
              filesystem-bound and not representative of model compute.
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

from compare_reparameterization import (
    load_training_model,
    main_logits,
    make_weaving_tta_batches,
)


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


def measure_latency_weaving8(
    model_cpu: nn.Module,
    height: int,
    width: int,
    device: torch.device,
    warmup: int,
    iters: int,
    use_data_parallel: bool,
) -> Dict[str, float]:
    """Time the released test_one_img_from_path_8 pattern: 4x forward(B=2).

    Deliberately NOT wrapped in torch.inference_mode()/no_grad(): the
    released test code does not disable autograd either (see the docstring
    of compare_reparameterization.weaving_memory_once for the same choice).
    """
    if device.type != "cuda":
        raise RuntimeError("weaving8 latency benchmark requires CUDA")

    model = model_cpu.to(device)
    model.eval()
    if use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[device.index or 0])

    def one_pass() -> None:
        img1, img2, img3, img4 = make_weaving_tta_batches(height, width, device)
        maska = main_logits(model(img1)).squeeze().cpu().data.numpy()
        maskb = main_logits(model(img2)).squeeze().cpu().data.numpy()
        maskc = main_logits(model(img3)).squeeze().cpu().data.numpy()
        maskd = main_logits(model(img4)).squeeze().cpu().data.numpy()
        try:
            mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
            _ = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        except Exception:
            # Output shape may differ from WeavingUnet (two-class logits);
            # the four forwards are already timed, so this does not
            # invalidate the measurement.
            pass

    for _ in range(warmup):
        one_pass()
    torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        one_pass()
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    model_cpu.to("cpu")
    del model
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
        "--protocol",
        choices=("single", "weaving8"),
        default="weaving8",
        help=(
            "single: one B=1 forward, no TTA (matches the GMAC path, NOT "
            "comparable to WeavingUnet's published Table-5 inference time). "
            "weaving8: the released test_one_img_from_path_8 pattern (4x "
            "forward at B=2, 8 TTA views/image) -- use this for a fair "
            "comparison against Table~\\ref{tab:sota}."
        ),
    )
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weaving8 only: match released code, which wraps the model in DataParallel",
    )
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
    print(f"LATENCY BENCHMARK (protocol={args.protocol})")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Weights    : {args.weights} | epoch={checkpoint.get('epoch')}")
    print(f"GPU        : {torch.cuda.get_device_name(device)}")
    if args.protocol == "single":
        print(f"Input      : 1x3x{args.height}x{args.width} | FP32 | eval mode | no TTA")
    else:
        print(
            f"Input      : 2x3x{args.height}x{args.width} x4 forwards "
            f"(8 TTA views/image) | FP32 | eval mode | "
            f"DataParallel={args.data_parallel}; autograd=ON"
        )
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
        "protocol": args.protocol,
        "gpu": torch.cuda.get_device_name(device),
        "input": [1, 3, args.height, args.width],
        "forms": {},
    }

    for label, model in selected:
        if args.protocol == "single":
            stats = measure_latency(model, args.height, args.width, device, args.warmup, args.iters)
        else:
            stats = measure_latency_weaving8(
                model, args.height, args.width, device, args.warmup, args.iters, args.data_parallel
            )
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
    if args.protocol == "weaving8":
        print(
            "For Table 5, use DEPLOY weaving8 latency (mean, ms/image) -- this "
            "matches WeavingUnet's released test_one_img_from_path_8 timing loop."
        )
    else:
        print(
            "single-protocol latency is NOT comparable to WeavingUnet's "
            "published Table-5 inference time; use --protocol weaving8 for that."
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON : {out}")


if __name__ == "__main__":
    main()
