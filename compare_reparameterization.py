"""WeavingUnet-style resource benchmark for DualBranchRoadNet.

Purpose
-------
Measure the three Table-5-style resource metrics without traversing a dataset:
  1) Parameters (M)
  2) Compute at 1024x1024 (GMACs and 2x-GFLOPs, both reported)
  3) Peak GPU memory during an 8-view TTA inference path patterned after the
     released WeavingUnet testmassa.py/testdg.py code.

The Weaving-style memory path intentionally uses:
  - FP32 (no AMP)
  - NCHW
  - 1024x1024 input
  - four CUDA tensors resident at once, each with batch size 2
  - four sequential forward passes = 8 transformed views/original image
  - GPU->CPU conversion after each forward
  - autograd enabled, matching the released test code (no no_grad/inference_mode)
  - optional single-GPU DataParallel wrapper (enabled by default)

No dataset is read and no segmentation metrics are computed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modeling.model import build_model


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in (
            "best_fixed_road_iou.pt",
            "best_fixed_iou.pt",
            "best.pt",
            "best_calibrated_road_iou.pt",
            "last.pt",
        ):
            candidate = path / name
            if candidate.is_file():
                return candidate
        candidates = sorted(path.rglob("*.pt")) + sorted(path.rglob("*.pth"))
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f"Checkpoint not found or ambiguous: {path}")


def clean_state_dict(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def main_logits(output: Any) -> Tensor:
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("Dictionary model output has no 'logits' entry")
        output = output["logits"]
    if isinstance(output, tuple):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output


def load_training_model(
    checkpoint_path: str | Path,
    weights: str,
) -> Tuple[nn.Module, dict, Path]:
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint")

    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise KeyError("Checkpoint has no dictionary 'args' entry")

    model_args = dict(saved_args)
    model_args["imagenet_pretrained"] = False
    model_args["encoder_weights_path"] = None
    model_args["deploy"] = False
    model = build_model(argparse.Namespace(**model_args))

    state = checkpoint.get(weights)
    if not isinstance(state, dict):
        fallback = "model" if weights == "ema" else "ema"
        state = checkpoint.get(fallback, checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise KeyError(f"No usable '{weights}', model, ema, or state_dict weights")

    model.load_state_dict(clean_state_dict(state), strict=True)
    return model, checkpoint, checkpoint_path


def count_rep_modules(model: nn.Module) -> Dict[str, int]:
    result = {"rep_total": 0, "training_form": 0, "deploy_form": 0}
    for module in model.modules():
        if module.__class__.__name__ not in {"RepVGGBlock", "RepDepthwiseBlock"}:
            continue
        result["rep_total"] += 1
        key = "deploy_form" if bool(getattr(module, "deploy", False)) else "training_form"
        result[key] += 1
    return result


def parameter_statistics(model: nn.Module) -> Dict[str, float]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = sum(x.numel() * x.element_size() for x in model.state_dict().values())
    return {
        "parameters": int(params),
        "parameters_m": float(params) / 1e6,
        "trainable_parameters": int(trainable),
        "state_bytes": int(state_bytes),
        "state_mib": float(state_bytes) / (1024.0**2),
    }


def _register_mac_hooks(
    model: nn.Module,
) -> Tuple[Dict[str, int], List[torch.utils.hooks.RemovableHandle]]:
    """Count MACs for Conv2d/ConvTranspose2d/Linear in one forward.

    We report BOTH conventions:
      - THOP-style operation count: 1 MAC reported as 1 "FLOP" in many papers
      - arithmetic FLOPs:          1 MAC = 2 FLOPs (multiply + add)

    This avoids silently choosing a convention that the WeavingUnet paper does
    not explicitly document in the supplied resource/table code.
    """
    counter = {"macs": 0}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def conv2d_hook(module: nn.Conv2d, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        if not torch.is_tensor(output):
            return
        kh, kw = module.kernel_size
        ops_per_output = kh * kw * module.in_channels // module.groups
        counter["macs"] += int(output.numel()) * int(ops_per_output)

    def convtranspose_hook(
        module: nn.ConvTranspose2d, inputs: Tuple[Tensor, ...], _output: Tensor
    ) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0]
        kh, kw = module.kernel_size
        # Each input element contributes to Cout/groups * kh * kw output MACs.
        ops_per_input = kh * kw * module.out_channels // module.groups
        counter["macs"] += int(x.numel()) * int(ops_per_input)

    def linear_hook(module: nn.Linear, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        if torch.is_tensor(output):
            counter["macs"] += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(convtranspose_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    return counter, handles


@torch.inference_mode()
def profile_compute(
    model_cpu: nn.Module,
    height: int,
    width: int,
    device: torch.device,
) -> Dict[str, float]:
    """One FP32 B=1 forward at 1024x1024; no TTA."""
    model = model_cpu.to(device).eval()
    x = torch.randn(1, 3, height, width, device=device, dtype=torch.float32)

    counter, handles = _register_mac_hooks(model)
    try:
        main_logits(model(x))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    macs = float(counter["macs"])
    model.to("cpu")
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "macs": macs,
        "gmacs": macs / 1e9,
        # Some papers/profilers label MACs directly as FLOPs.
        "table_style_gflops_if_1mac_eq_1flop": macs / 1e9,
        # Mathematical multiply+add convention.
        "arithmetic_gflops_if_1mac_eq_2flops": 2.0 * macs / 1e9,
    }


def make_weaving_tta_batches(
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mimic released WeavingUnet test_one_img_from_path_8 preprocessing.

    A synthetic uint8 image replaces cv2.imread/resize because GPU memory is the
    target metric and the table fixes the inference image size to 1024x1024.
    """
    rng = np.random.default_rng(3407)
    img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img90 = np.array(np.rot90(img))

    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.array(img1)[:, :, ::-1]
    img4 = np.array(img2)[:, :, ::-1]

    batches = []
    for arr in (img1, img2, img3, img4):
        arr = arr.transpose(0, 3, 1, 2)
        arr = np.array(arr, np.float32) / 255.0 * 3.2 - 1.6
        # torch.Tensor(...) is FP32, matching the released test code.
        batches.append(torch.Tensor(arr).to(device))
    return batches[0], batches[1], batches[2], batches[3]


def weaving_memory_once(
    model_cpu: nn.Module,
    height: int,
    width: int,
    device: torch.device,
    use_data_parallel: bool,
    literal_train_mode: bool,
) -> Dict[str, float]:
    """Peak memory using the released 8-view / four-B2-forward pattern.

    IMPORTANT: intentionally NOT decorated with no_grad/inference_mode. The
    released test code does not disable autograd.
    """
    if device.type != "cuda":
        raise RuntimeError("Weaving-style GPU-memory benchmark requires CUDA")

    gc.collect()
    torch.cuda.empty_cache()

    model = model_cpu.to(device)
    # Paper calls this the inference stage, so eval() is the sensible default.
    # --literal-test-code leaves the model in train mode because the released
    # bottom-level test loop calls test_one_img_from_path_8 directly.
    if literal_train_mode:
        model.train()
    else:
        model.eval()

    if use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[device.index or 0])

    # Match the released code: all four B=2 FP32 CUDA input tensors exist before
    # the first forward pass.
    img1, img2, img3, img4 = make_weaving_tta_batches(height, width, device)

    torch.cuda.synchronize(device)
    baseline_alloc = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    # Four sequential B=2 forwards -> 8 TTA views/original image.
    # GPU->CPU .data.numpy() after each forward follows their released test code.
    maska = main_logits(model(img1)).squeeze().cpu().data.numpy()
    maskb = main_logits(model(img2)).squeeze().cpu().data.numpy()
    maskc = main_logits(model(img3)).squeeze().cpu().data.numpy()
    maskd = main_logits(model(img4)).squeeze().cpu().data.numpy()

    # Reproduce their CPU-side merge pattern sufficiently to keep the full
    # inference path faithful. This has no material effect on CUDA peak memory.
    try:
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        _ = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
    except Exception:
        # Output shape may differ from WeavingUnet (e.g. two-class logits). Peak
        # CUDA memory has already been measured, so do not invalidate the result.
        pass

    torch.cuda.synchronize(device)
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    del maska, maskb, maskc, maskd
    del img1, img2, img3, img4
    # .to(device) is in-place for modules; explicitly return the caller-owned
    # model to CPU so a second form/repeat does not contaminate the next peak.
    model_cpu.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "baseline_allocated_bytes": baseline_alloc,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_alloc / (1024.0**3),
        "peak_reserved_gib": peak_reserved / (1024.0**3),
        "incremental_peak_allocated_gib": (peak_alloc - baseline_alloc) / (1024.0**3),
    }


def weaving_memory_profile(
    model_cpu: nn.Module,
    height: int,
    width: int,
    device: torch.device,
    repeats: int,
    use_data_parallel: bool,
    literal_train_mode: bool,
) -> Dict[str, float]:
    rows = [
        weaving_memory_once(
            model_cpu,
            height,
            width,
            device,
            use_data_parallel,
            literal_train_mode,
        )
        for _ in range(repeats)
    ]
    peak_alloc = np.asarray([r["peak_allocated_gib"] for r in rows], dtype=np.float64)
    peak_reserved = np.asarray([r["peak_reserved_gib"] for r in rows], dtype=np.float64)
    return {
        "repeats": repeats,
        "peak_allocated_gib_mean": float(peak_alloc.mean()),
        "peak_allocated_gib_max": float(peak_alloc.max()),
        "peak_allocated_gib_min": float(peak_alloc.min()),
        "peak_allocated_gib_std": float(peak_alloc.std()),
        "peak_reserved_gib_mean": float(peak_reserved.mean()),
        "peak_reserved_gib_max": float(peak_reserved.max()),
        "last_baseline_allocated_gib": rows[-1]["baseline_allocated_bytes"] / (1024.0**3),
        "last_incremental_peak_allocated_gib": rows[-1]["incremental_peak_allocated_gib"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Table-5-style Params/FLOPs/Mem benchmark patterned after WeavingUnet",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match released WeavingUnet test code, which wraps the model in DataParallel",
    )
    parser.add_argument(
        "--literal-test-code",
        action="store_true",
        help=(
            "Leave model in train mode to mimic the bottom-level released test loop literally. "
            "Default uses eval(), which is the appropriate paper inference-stage setting."
        ),
    )
    parser.add_argument(
        "--form",
        choices=("deploy", "multi", "both"),
        default="deploy",
        help="Use deploy for the final model row in a resource-comparison table",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def print_form(label: str, stats: Dict[str, Any]) -> None:
    p = stats["parameters"]
    c = stats["compute"]
    m = stats["memory"]
    print("-" * 88)
    print(label)
    print(f"Params       : {p['parameters_m']:.3f} M  ({p['parameters']:,} parameters)")
    print(f"State size   : {p['state_mib']:.2f} MiB")
    print(f"Compute      : {c['gmacs']:.3f} GMACs @ 1x3x1024x1024")
    print(
        f"Table-style  : {c['table_style_gflops_if_1mac_eq_1flop']:.3f} G "
        "if profiler/paper labels 1 MAC as 1 FLOP"
    )
    print(
        f"Arithmetic   : {c['arithmetic_gflops_if_1mac_eq_2flops']:.3f} GFLOPs "
        "under 1 MAC = 2 FLOPs"
    )
    print(
        f"Peak Mem     : {m['peak_allocated_gib_mean']:.3f} GiB mean | "
        f"max={m['peak_allocated_gib_max']:.3f} GiB | std={m['peak_allocated_gib_std']:.3f}"
    )
    print(
        f"Peak reserved: {m['peak_reserved_gib_mean']:.3f} GiB mean | "
        f"max={m['peak_reserved_gib_max']:.3f} GiB"
    )


def main() -> None:
    args = parse_args()
    if min(args.height, args.width, args.memory_repeats) < 1:
        raise ValueError("height, width and memory-repeats must be positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for the memory benchmark")

    training_model, checkpoint, checkpoint_path = load_training_model(args.ckpt, args.weights)
    training_model.eval()

    deploy_model = copy.deepcopy(training_model)
    if not hasattr(deploy_model, "switch_to_deploy"):
        raise AttributeError("Model has no switch_to_deploy() method")
    deploy_model.switch_to_deploy()
    deploy_model.eval()

    rep_train = count_rep_modules(training_model)
    rep_deploy = count_rep_modules(deploy_model)
    if rep_deploy["training_form"] != 0:
        raise RuntimeError(f"Not every rep block was fused: {rep_deploy}")

    print("=" * 88)
    print("WEAVINGUNET TABLE-5-STYLE RESOURCE BENCHMARK")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Weights    : {args.weights} | epoch={checkpoint.get('epoch')}")
    print(f"GPU        : {torch.cuda.get_device_name(device)}")
    print(f"Input      : 1024x1024 | FP32 | NCHW")
    print("FLOPs path : one B=1 forward, no TTA")
    print(
        "Mem path   : 8-view TTA = 4 sequential forwards x B2; FP32; "
        f"DataParallel={args.data_parallel}; autograd=ON"
    )
    print(
        "Mode       : "
        + ("literal released test-loop train mode" if args.literal_test_code else "eval mode")
    )
    print(f"Rep blocks : multi={rep_train['training_form']} | deploy={rep_deploy['deploy_form']}")

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
        "memory_protocol": {
            "precision": "FP32",
            "tta_views": 8,
            "forward_passes": 4,
            "batch_per_forward": 2,
            "data_parallel": bool(args.data_parallel),
            "autograd_enabled": True,
            "model_mode": "train_literal" if args.literal_test_code else "eval",
        },
        "forms": {},
    }

    for label, model in selected:
        stats = {
            "parameters": parameter_statistics(model),
            "compute": profile_compute(model, args.height, args.width, device),
            "memory": weaving_memory_profile(
                model,
                args.height,
                args.width,
                device,
                args.memory_repeats,
                args.data_parallel,
                args.literal_test_code,
            ),
        }
        result["forms"][label] = stats
        print_form(label, stats)

    print("=" * 88)
    print("For Table 5, use DEPLOY Params + the clearly stated FLOPs convention + Peak allocated Mem.")
    print("Do NOT use the old FP16/B1 memory number for comparison with WeavingUnet.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON : {out}")


if __name__ == "__main__":
    main()
