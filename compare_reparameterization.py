"""One-file comparison of multi-branch and reparameterized inference.

Loads one DualBranchRoadNet checkpoint, builds both forms, checks FP32 output
equivalence, benchmarks FLOPs/latency/throughput/VRAM, and evaluates both forms on
the same Massachusetts subset with native-resolution sliding-window inference.
Models run sequentially on CUDA to keep memory use suitable for a Kaggle T4.

This measures deployment equivalence and efficiency. It does not compare with
a different single-branch architecture trained from scratch.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modeling.model import build_model


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in (
            "best_fixed_road_iou.pt", "best_fixed_iou.pt", "best.pt",
            "best_calibrated_road_iou.pt", "last.pt",
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
    checkpoint_path: str | Path, weights: str,
) -> Tuple[nn.Module, dict, Path]:
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
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
    return model.eval(), checkpoint, checkpoint_path


def count_rep_modules(model: nn.Module) -> Dict[str, int]:
    result = {"rep_total": 0, "training_form": 0, "deploy_form": 0}
    for module in model.modules():
        if module.__class__.__name__ not in {"RepVGGBlock", "RepDepthwiseBlock"}:
            continue
        result["rep_total"] += 1
        key = "deploy_form" if bool(getattr(module, "deploy", False)) else "training_form"
        result[key] += 1
    return result


def parameter_statistics(model: nn.Module) -> Dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameters": sum(x.numel() for x in parameters),
        "trainable_parameters": sum(x.numel() for x in parameters if x.requires_grad),
        "state_bytes": sum(
            x.numel() * x.element_size() for x in model.state_dict().values()
        ),
    }


def amp_settings(mode: str, device: torch.device) -> Tuple[bool, torch.dtype]:
    if device.type != "cuda" or mode == "none":
        return False, torch.float32
    if mode == "bfloat16":
        return True, torch.bfloat16
    if mode == "auto" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def register_mac_hooks(
    model: nn.Module,
) -> Tuple[Dict[str, int], List[torch.utils.hooks.RemovableHandle]]:
    """Count Conv/Linear MACs during one forward pass without extra packages.

    The reported compute follows the common convention ``1 MAC = 2 FLOPs``.
    BatchNorm, activations, interpolation, pooling, and elementwise operations
    are intentionally excluded so the number is comparable to most CNN FLOPs
    profilers and directly exposes the effect of fusing convolution branches.
    """
    counter = {"macs": 0}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        if not torch.is_tensor(output):
            return
        kernel_h, kernel_w = module.kernel_size
        multiplications_per_output = (
            kernel_h * kernel_w * module.in_channels // module.groups
        )
        counter["macs"] += int(output.numel()) * multiplications_per_output

    def linear_hook(module: nn.Linear, _inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        if torch.is_tensor(output):
            counter["macs"] += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    return counter, handles


@torch.inference_mode()
def profile_model(
    model_cpu: nn.Module,
    input_cpu: Tensor,
    device: torch.device,
    amp: str,
    channels_last: bool,
    warmup: int,
    repeats: int,
) -> Tuple[Tensor, Dict[str, float]]:
    model = model_cpu.to(device).eval()
    x = input_cpu.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)

    mac_counter, mac_handles = register_mac_hooks(model)
    try:
        fp32_output = main_logits(model(x)).float().cpu()
    finally:
        for handle in mac_handles:
            handle.remove()
    macs_per_image = float(mac_counter["macs"]) / float(input_cpu.shape[0])
    enabled, dtype = amp_settings(amp, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            main_logits(model(x))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            with torch.autocast(device.type, dtype=dtype, enabled=enabled):
                main_logits(model(x))
        end.record()
        torch.cuda.synchronize(device)
        total_ms = float(start.elapsed_time(end))
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        started = time.perf_counter()
        for _ in range(repeats):
            main_logits(model(x))
        total_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = 0
        peak_reserved_bytes = 0

    batch_size = int(input_cpu.shape[0])
    batch_latency_ms = total_ms / repeats
    latency_ms_per_image = batch_latency_ms / max(batch_size, 1)
    result = {
        "benchmark_batch_size": float(batch_size),
        "benchmark_total_ms": total_ms,
        "macs_per_image": macs_per_image,
        "flops_per_image": 2.0 * macs_per_image,
        "gmacs_per_image": macs_per_image / 1e9,
        "gflops_per_image": 2.0 * macs_per_image / 1e9,
        # latency_ms is retained for compatibility and means latency per batch.
        "latency_ms": batch_latency_ms,
        "latency_ms_per_batch": batch_latency_ms,
        "latency_ms_per_image": latency_ms_per_image,
        "throughput_images_per_second": 1000.0 / max(latency_ms_per_image, 1e-12),
        "peak_allocated_bytes": peak_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }
    model.to("cpu")
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fp32_output, result


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in (
        "_image", "_images", "_img", "_sat", "_mask", "_masks",
        "_gt", "_label", "_labels",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def index_files(folder: str | Path) -> Dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No supported images found in {folder}")
    result: Dict[str, Path] = {}
    for path in files:
        key = sample_key(path)
        if key in result:
            raise RuntimeError(f"Duplicate sample key '{key}': {result[key]} and {path}")
        result[key] = path
    return result


def pairs_from_list(
    image_dir: str | Path, mask_dir: str | Path, list_path: str | Path,
) -> List[Tuple[Path, Path]]:
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Split txt not found: {list_path}")
    images, masks = index_files(image_dir), index_files(mask_dir)
    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    seen: set[str] = set()
    with list_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entry = line.replace(",", " ").split()[0]
            key = sample_key(Path(entry))
            if key in seen:
                continue
            seen.add(key)
            image_path, mask_path = images.get(key), masks.get(key)
            if image_path is None or mask_path is None:
                missing.append(entry)
            else:
                pairs.append((image_path, mask_path))
    if missing:
        raise RuntimeError(
            f"Could not pair {len(missing)} entries; first missing: {missing[:10]}"
        )
    if not pairs:
        raise RuntimeError(f"No pairs resolved from {list_path}")
    return pairs


def select_subset(
    pairs: Sequence[Tuple[Path, Path]], subset: str, val_count: int,
) -> List[Tuple[Path, Path]]:
    if subset == "val61":
        return list(pairs[:val_count])
    if subset == "test117":
        return list(pairs[val_count:])
    if subset in {"all178", "custom"}:
        return list(pairs)
    raise ValueError(f"Unsupported subset: {subset}")


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    threshold = 0 if int(mask.max(initial=0)) <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def image_to_tensor(image: np.ndarray) -> Tensor:
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)


def sliding_positions(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def hann_weight(window: int, device: torch.device) -> Tensor:
    axis = torch.hann_window(
        window, periodic=False, dtype=torch.float32, device=device
    ).clamp_min_(0.05)
    return (axis[:, None] * axis[None, :]).unsqueeze(0).unsqueeze(0)


def road_probability(logits: Tensor) -> Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected 4-D logits, got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return logits.sigmoid()
    if logits.shape[1] == 2:
        return logits.softmax(dim=1)[:, 1:2]
    raise ValueError(f"Expected one or two output channels, got {logits.shape[1]}")


@torch.inference_mode()
def sliding_logits(
    model: nn.Module,
    x: Tensor,
    window: int,
    stride: int,
    tile_batch_size: int,
    amp: str,
    channels_last: bool,
) -> Tensor:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Expected x with shape [1,C,H,W]")
    device = next(model.parameters()).device
    original_h, original_w = x.shape[-2:]
    pad_h, pad_w = max(0, window - original_h), max(0, window - original_w)
    if pad_h or pad_w:
        mode = "reflect" if min(original_h, original_w) > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    height, width = x.shape[-2:]
    coords = [
        (y, xx)
        for y in sliding_positions(height, window, stride)
        for xx in sliding_positions(width, window, stride)
    ]
    accumulator: Tensor | None = None
    normalizer = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    weight = hann_weight(window, device)
    enabled, dtype = amp_settings(amp, device)

    for start in range(0, len(coords), tile_batch_size):
        batch_coords = coords[start : start + tile_batch_size]
        tiles = torch.cat(
            [x[:, :, y:y + window, xx:xx + window] for y, xx in batch_coords], dim=0
        ).to(device, non_blocking=True)
        if channels_last:
            tiles = tiles.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            logits = main_logits(model(tiles)).float()
        if accumulator is None:
            accumulator = torch.zeros(
                (1, logits.shape[1], height, width), device=device, dtype=torch.float32
            )
        for index, (y, xx) in enumerate(batch_coords):
            accumulator[:, :, y:y + window, xx:xx + window] += (
                logits[index:index + 1] * weight
            )
            normalizer[:, :, y:y + window, xx:xx + window] += weight
    assert accumulator is not None
    return (accumulator / normalizer.clamp_min_(1e-6))[:, :, :original_h, :original_w]


def tta_tags(mode: str) -> Tuple[str, ...]:
    if mode == "none":
        return ("r0",)
    if mode == "flip4":
        return ("r0", "fr0", "fr2", "r2")
    if mode == "d4":
        return ("r0", "r1", "r2", "r3", "fr0", "fr1", "fr2", "fr3")
    raise ValueError(f"Unsupported TTA mode: {mode}")


def apply_tta(tensor: Tensor, tag: str) -> Tensor:
    output = torch.rot90(tensor, int(tag[-1]), dims=(-2, -1))
    return torch.flip(output, dims=(-1,)) if tag.startswith("f") else output


def invert_tta(tensor: Tensor, tag: str) -> Tensor:
    output = torch.flip(tensor, dims=(-1,)) if tag.startswith("f") else tensor
    return torch.rot90(output, -int(tag[-1]), dims=(-2, -1))


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image: np.ndarray,
    window: int,
    stride: int,
    tta_mode: str,
    tta_merge: str,
    amp: str,
    tile_batch_size: int,
    channels_last: bool,
) -> np.ndarray:
    x = image_to_tensor(image)
    total_probability: Tensor | None = None
    total_logits: Tensor | None = None
    tags = tta_tags(tta_mode)
    for tag in tags:
        logits = sliding_logits(
            model, apply_tta(x, tag), window, stride, tile_batch_size,
            amp, channels_last,
        )
        logits = invert_tta(logits, tag)
        if tta_merge == "probabilities":
            probability = road_probability(logits)
            total_probability = (
                probability if total_probability is None else total_probability + probability
            )
        else:
            total_logits = logits if total_logits is None else total_logits + logits
    if tta_merge == "probabilities":
        assert total_probability is not None
        return (total_probability / len(tags))[0, 0].cpu().numpy()
    assert total_logits is not None
    return road_probability(total_logits / len(tags))[0, 0].cpu().numpy()


def confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return (
        int(np.logical_and(pred, gt).sum()),
        int(np.logical_and(pred, np.logical_not(gt)).sum()),
        int(np.logical_and(np.logical_not(pred), gt).sum()),
        int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()),
    )


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "background_iou": background_iou,
        "miou": 0.5 * (iou + background_iou),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


@torch.inference_mode()
def evaluate_model(
    label: str,
    model_cpu: nn.Module,
    pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    threshold: float,
    window: int,
    stride: int,
    tta_mode: str,
    tta_merge: str,
    amp: str,
    tile_batch_size: int,
    channels_last: bool,
) -> Dict[str, float]:
    model = model_cpu.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    pooled = [0, 0, 0, 0]
    image_f1: List[float] = []
    image_iou: List[float] = []
    inference_times: List[float] = []
    io_seconds = 0.0
    total_pixels = 0
    total_tiles = 0
    tta_views = len(tta_tags(tta_mode))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        io_started = time.perf_counter()
        image, gt = read_rgb(image_path), read_binary_mask(mask_path)
        io_seconds += time.perf_counter() - io_started

        tiles_per_view = (
            len(sliding_positions(image.shape[0], window, stride))
            * len(sliding_positions(image.shape[1], window, stride))
        )
        total_tiles += tiles_per_view * tta_views

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        probability = predict_image(
            model, image, window, stride, tta_mode, tta_merge,
            amp, tile_batch_size, channels_last,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds_this_image = time.perf_counter() - inference_started
        inference_times.append(inference_seconds_this_image)

        height = min(gt.shape[0], probability.shape[0])
        width = min(gt.shape[1], probability.shape[1])
        total_pixels += height * width
        values = confusion(
            probability[:height, :width] >= threshold, gt[:height, :width]
        )
        for position, value in enumerate(values):
            pooled[position] += value
        image_metrics = metrics_from_counts(*values)
        image_f1.append(image_metrics["f1"])
        image_iou.append(image_metrics["iou"])
        running_elapsed = max(time.perf_counter() - started, 1e-12)
        print(
            f"\r[{label}] {index:3d}/{len(pairs)}  {image_path.name}  "
            f"P={image_metrics['precision']:.4f} R={image_metrics['recall']:.4f} "
            f"F1={image_metrics['f1']:.4f} IoU={image_metrics['iou']:.4f} "
            f"infer={1000.0 * inference_seconds_this_image:.1f}ms "
            f"e2e={index / running_elapsed:.2f} img/s",
            end="", flush=True,
        )
    print()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_bytes = 0
        peak_reserved_bytes = 0
    elapsed = time.perf_counter() - started
    inference_seconds = float(sum(inference_times))
    inference_ms = np.asarray(inference_times, dtype=np.float64) * 1000.0
    result: Dict[str, float] = {
        **metrics_from_counts(*pooled),
        "mean_image_f1": float(np.mean(image_f1)),
        "mean_image_iou": float(np.mean(image_iou)),
        "elapsed_seconds": elapsed,
        # Backward-compatible alias: this is end-to-end dataset throughput.
        "images_per_second": len(pairs) / max(elapsed, 1e-12),
        "end_to_end_images_per_second": len(pairs) / max(elapsed, 1e-12),
        "end_to_end_ms_per_image": 1000.0 * elapsed / max(len(pairs), 1),
        "inference_seconds": inference_seconds,
        "inference_images_per_second": len(pairs) / max(inference_seconds, 1e-12),
        "inference_ms_per_image_mean": float(inference_ms.mean()),
        "inference_ms_per_image_median": float(np.median(inference_ms)),
        "inference_ms_per_image_p95": float(np.percentile(inference_ms, 95)),
        "inference_ms_per_image_min": float(inference_ms.min()),
        "inference_ms_per_image_max": float(inference_ms.max()),
        "inference_ms_per_image_std": float(inference_ms.std()),
        "io_seconds": io_seconds,
        "total_pixels": float(total_pixels),
        "megapixels_per_second": (total_pixels / 1e6) / max(inference_seconds, 1e-12),
        "tta_views": float(tta_views),
        "tiles_total": float(total_tiles),
        "tiles_per_second": total_tiles / max(inference_seconds, 1e-12),
        "peak_allocated_bytes": float(peak_bytes),
        "peak_reserved_bytes": float(peak_reserved_bytes),
    }
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="One-run multi-branch versus deploy comparison",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--thr", type=float, default=0.50)
    parser.add_argument(
        "--subset", choices=("val61", "test117", "all178", "custom"),
        default="val61",
    )
    parser.add_argument("--val-count", type=int, default=61)
    parser.add_argument("--limit", type=int, default=None)
    root = "/kaggle/input/datasets/datnguyentien204/massachu/massachusets"
    parser.add_argument("--data-root", default=root)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--tta-mode", choices=("none", "flip4", "d4"), default="none")
    parser.add_argument(
        "--tta-merge", choices=("probabilities", "logits"),
        default="probabilities",
    )
    parser.add_argument(
        "--amp", choices=("auto", "float16", "bfloat16", "none"),
        default="float16",
    )
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--height", type=int, default=1024, help="benchmark input height")
    parser.add_argument("--width", type=int, default=1024, help="benchmark input width")
    parser.add_argument("--batch-size", type=int, default=1, help="benchmark batch size")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_benchmark and args.skip_eval:
        raise ValueError("Cannot combine --skip-benchmark and --skip-eval")
    if not 0.0 <= args.thr <= 1.0:
        raise ValueError("--thr must be in [0,1]")
    if min(args.height, args.width, args.batch_size, args.repeats) < 1:
        raise ValueError("height, width, batch-size, and repeats must be positive")
    if args.warmup < 0 or args.tile_batch_size < 1:
        raise ValueError("warmup must be nonnegative and tile-batch-size positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    training_model, checkpoint, checkpoint_path = load_training_model(
        args.ckpt, args.weights
    )
    training_rep = count_rep_modules(training_model)
    training_stats = parameter_statistics(training_model)
    deploy_model = copy.deepcopy(training_model).eval()
    if not hasattr(deploy_model, "switch_to_deploy"):
        raise AttributeError("Model has no switch_to_deploy() method")
    deploy_model.switch_to_deploy()
    deploy_model.eval()
    deploy_rep = count_rep_modules(deploy_model)
    deploy_stats = parameter_statistics(deploy_model)
    if deploy_rep["training_form"] != 0:
        raise RuntimeError(f"Not every rep block was fused: {deploy_rep}")

    saved_args = checkpoint["args"]
    window = args.window or int(saved_args.get("val_tile_size", 1024))
    if args.stride is None:
        overlap = int(saved_args.get("val_overlap", window // 2))
        stride = window - overlap
    else:
        overlap, stride = window - args.stride, args.stride
    if not 1 <= stride <= window:
        raise ValueError("Resolved stride must satisfy 1 <= stride <= window")

    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "device": str(device),
        "training_form": {**training_rep, **training_stats},
        "deploy_form": {**deploy_rep, **deploy_stats},
    }

    print("=" * 88)
    print("MULTI-BRANCH vs REPARAMETERIZED DEPLOY — ONE-RUN COMPARISON")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Weights    : {args.weights} | epoch={checkpoint.get('epoch')}")
    print(
        f"Device     : {device} | AMP={args.amp} | "
        f"memory={'NHWC' if args.channels_last else 'NCHW'}"
    )
    print(
        f"Rep blocks : training={training_rep['training_form']} | "
        f"fused={deploy_rep['deploy_form']}"
    )

    equivalent = True
    if not args.skip_benchmark:
        input_cpu = torch.randn(
            args.batch_size, 3, args.height, args.width, dtype=torch.float32
        )
        training_output, training_profile = profile_model(
            training_model, input_cpu, device, args.amp, args.channels_last,
            args.warmup, args.repeats,
        )
        deploy_output, deploy_profile = profile_model(
            deploy_model, input_cpu, device, args.amp, args.channels_last,
            args.warmup, args.repeats,
        )
        difference = (training_output - deploy_output).abs()
        max_abs, mean_abs = float(difference.max()), float(difference.mean())
        equivalent = bool(torch.allclose(
            training_output, deploy_output, atol=args.atol, rtol=args.rtol
        ))
        speedup = training_profile["latency_ms"] / deploy_profile["latency_ms"]
        flops_reduction = 1.0 - (
            deploy_profile["flops_per_image"]
            / training_profile["flops_per_image"]
        )
        result["training_form"].update(training_profile)
        result["deploy_form"].update(deploy_profile)
        result["equivalence"] = {
            "max_abs_output_error_fp32": max_abs,
            "mean_abs_output_error_fp32": mean_abs,
            "allclose": equivalent,
            "atol": args.atol,
            "rtol": args.rtol,
        }
        result["benchmark_speedup"] = speedup
        result["flops_reduction_fraction"] = flops_reduction

        print("-" * 88)
        print("SYNTHETIC TILE BENCHMARK")
        print(
            f"Input      : {tuple(input_cpu.shape)} | warmup={args.warmup} | "
            f"repeats={args.repeats}"
        )
        print(
            f"Parameters : train={training_stats['parameters']:,} | "
            f"deploy={deploy_stats['parameters']:,} | "
            f"change={deploy_stats['parameters'] - training_stats['parameters']:+,}"
        )
        print(
            f"State size : train={mib(training_stats['state_bytes']):.2f} MiB | "
            f"deploy={mib(deploy_stats['state_bytes']):.2f} MiB"
        )
        print(
            f"Compute    : train={training_profile['gmacs_per_image']:.3f} GMACs / "
            f"{training_profile['gflops_per_image']:.3f} GFLOPs | "
            f"deploy={deploy_profile['gmacs_per_image']:.3f} GMACs / "
            f"{deploy_profile['gflops_per_image']:.3f} GFLOPs"
        )
        print(
            f"FLOPs cut  : {100.0 * flops_reduction:.2f}% "
            f"(Conv/Linear only; 1 MAC = 2 FLOPs)"
        )
        print(
            f"Latency/batch: train={training_profile['latency_ms_per_batch']:.3f} ms | "
            f"deploy={deploy_profile['latency_ms_per_batch']:.3f} ms | speedup={speedup:.3f}x"
        )
        print(
            f"Latency/img  : train={training_profile['latency_ms_per_image']:.3f} ms | "
            f"deploy={deploy_profile['latency_ms_per_image']:.3f} ms"
        )
        print(
            f"Throughput   : train={training_profile['throughput_images_per_second']:.2f} img/s | "
            f"deploy={deploy_profile['throughput_images_per_second']:.2f} img/s"
        )
        print(
            f"Peak allocated: train={mib(training_profile['peak_allocated_bytes']):.2f} MiB | "
            f"deploy={mib(deploy_profile['peak_allocated_bytes']):.2f} MiB"
        )
        print(
            f"Peak reserved : train={mib(training_profile['peak_reserved_bytes']):.2f} MiB | "
            f"deploy={mib(deploy_profile['peak_reserved_bytes']):.2f} MiB"
        )
        print(
            f"FP32 error : max={max_abs:.8e} | mean={mean_abs:.8e} | "
            f"allclose={equivalent}"
        )

    if not args.skip_eval:
        data_root = Path(args.data_root)
        image_dir = Path(args.image_dir) if args.image_dir else data_root / "images"
        mask_dir = Path(args.mask_dir) if args.mask_dir else data_root / "labels"
        test_list = Path(args.test_list) if args.test_list else data_root / "test.txt"
        all_pairs = pairs_from_list(image_dir, mask_dir, test_list)
        pairs = select_subset(all_pairs, args.subset, args.val_count)
        if args.limit is not None:
            pairs = pairs[:args.limit]
        if not pairs:
            raise RuntimeError("Selected evaluation subset is empty")

        print("-" * 88)
        print("NATIVE-RESOLUTION DATASET EVALUATION")
        print(f"List       : {test_list} ({len(all_pairs)} total entries)")
        print(f"Subset     : {args.subset} | images={len(pairs)} | val_count={args.val_count}")
        print(f"Inference  : window={window} | stride={stride} | overlap={overlap}")
        print(
            f"Protocol   : Hann LOGIT blend | TTA={args.tta_mode}/{args.tta_merge} | "
            f"thr={args.thr:.2f}"
        )

        training_eval = evaluate_model(
            "MULTI", training_model, pairs, device, args.thr, window, stride,
            args.tta_mode, args.tta_merge, args.amp, args.tile_batch_size,
            args.channels_last,
        )
        deploy_eval = evaluate_model(
            "DEPLOY", deploy_model, pairs, device, args.thr, window, stride,
            args.tta_mode, args.tta_merge, args.amp, args.tile_batch_size,
            args.channels_last,
        )
        result["evaluation"] = {
            "subset": args.subset,
            "images": len(pairs),
            "threshold": args.thr,
            "window": window,
            "stride": stride,
            "overlap": overlap,
            "tta_mode": args.tta_mode,
            "tta_merge": args.tta_merge,
            "training_form": training_eval,
            "deploy_form": deploy_eval,
            "deploy_minus_training": {
                key: deploy_eval[key] - training_eval[key]
                for key in (
                    "precision", "recall", "f1", "iou", "miou", "accuracy",
                    "mean_image_f1", "mean_image_iou",
                )
            },
        }

        dataset_speedup = (
            deploy_eval["inference_images_per_second"]
            / max(training_eval["inference_images_per_second"], 1e-12)
        )
        result["evaluation"]["deploy_inference_speedup"] = dataset_speedup

        def print_full_eval(label: str, values: Dict[str, float]) -> None:
            print(f"{label} ACCURACY")
            print(
                f"  Precision={values['precision']:.6f} | Recall={values['recall']:.6f} | "
                f"F1/Dice={values['f1']:.6f}"
            )
            print(
                f"  Road IoU={values['iou']:.6f} | BG IoU={values['background_iou']:.6f} | "
                f"mIoU={values['miou']:.6f} | Accuracy={values['accuracy']:.6f}"
            )
            print(
                f"  Mean-image F1={values['mean_image_f1']:.6f} | "
                f"Mean-image IoU={values['mean_image_iou']:.6f}"
            )
            print(f"{label} SPEED / MEMORY")
            print(
                f"  End-to-end: {values['elapsed_seconds']:.3f}s total | "
                f"{values['end_to_end_ms_per_image']:.3f} ms/image | "
                f"{values['end_to_end_images_per_second']:.3f} images/s"
            )
            print(
                f"  Inference : {values['inference_seconds']:.3f}s total | "
                f"mean={values['inference_ms_per_image_mean']:.3f} ms | "
                f"median={values['inference_ms_per_image_median']:.3f} ms | "
                f"p95={values['inference_ms_per_image_p95']:.3f} ms"
            )
            print(
                f"              min={values['inference_ms_per_image_min']:.3f} ms | "
                f"max={values['inference_ms_per_image_max']:.3f} ms | "
                f"std={values['inference_ms_per_image_std']:.3f} ms | "
                f"{values['inference_images_per_second']:.3f} images/s"
            )
            print(
                f"  Workload  : TTA views={int(values['tta_views'])} | "
                f"tiles={int(values['tiles_total'])} | "
                f"{values['tiles_per_second']:.3f} tiles/s | "
                f"{values['megapixels_per_second']:.3f} MPix/s"
            )
            print(
                f"  I/O time  : {values['io_seconds']:.3f}s | "
                f"peak allocated={mib(values['peak_allocated_bytes']):.2f} MiB | "
                f"peak reserved={mib(values['peak_reserved_bytes']):.2f} MiB"
            )

        print("-" * 88)
        print("FINAL METRICS (same checkpoint, images, threshold and inference path)")
        print_full_eval("MULTI ", training_eval)
        print_full_eval("DEPLOY", deploy_eval)
        print("DELTA / SPEEDUP")
        print(
            f"  Precision={deploy_eval['precision'] - training_eval['precision']:+.8f} | "
            f"Recall={deploy_eval['recall'] - training_eval['recall']:+.8f} | "
            f"F1={deploy_eval['f1'] - training_eval['f1']:+.8f}"
        )
        print(
            f"  IoU={deploy_eval['iou'] - training_eval['iou']:+.8f} | "
            f"BG-IoU={deploy_eval['background_iou'] - training_eval['background_iou']:+.8f} | "
            f"mIoU={deploy_eval['miou'] - training_eval['miou']:+.8f} | "
            f"Accuracy={deploy_eval['accuracy'] - training_eval['accuracy']:+.8f}"
        )
        print(
            f"  Mean-F1={deploy_eval['mean_image_f1'] - training_eval['mean_image_f1']:+.8f} | "
            f"Mean-IoU={deploy_eval['mean_image_iou'] - training_eval['mean_image_iou']:+.8f}"
        )
        print(
            f"  Dataset inference speedup={dataset_speedup:.4f}x | "
            f"E2E throughput speedup="
            f"{deploy_eval['end_to_end_images_per_second'] / max(training_eval['end_to_end_images_per_second'], 1e-12):.4f}x"
        )

    print("=" * 88)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved JSON : {output_path}")

    if not equivalent:
        raise RuntimeError(
            "Fused output is outside tolerance; do not deploy until fusion is fixed"
        )


if __name__ == "__main__":
    main()
