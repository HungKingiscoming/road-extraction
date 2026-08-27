"""Trainer-matched native Massachusetts evaluation with optional TTA.

The default inference path reproduces train.py validation: native resolution,
ImageNet normalization, reflect padding, checkpoint-saved validation tile size
and overlap, and Hann-weighted LOGIT blending. Optional flip4 or D4 TTA applies
this complete path to each transformed full image and inverse-transforms the
complete blended map. TTA views can be merged as probabilities (recommended)
or logits.

``roadx3`` is a compatibility profile for the supplied roadx.infer code: pad
the full image to a stride multiple, use identity + horizontal + vertical
views, uniformly blend per-tile probabilities, inverse-transform each complete
probability canvas, and average the three canvases. Inverting the complete
canvas fixes the coordinate error caused by inverse-flipping each tile while
leaving it at the transformed tile coordinate.

The default ``test117`` subset excludes the first 61 test.txt entries used for
checkpoint selection.  Threshold search is intentionally restricted to
``val61`` so the final test set cannot be used for calibration by accident.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modeling.model import build_model


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in (
        "_image", "_images", "_img", "_sat",
        "_mask", "_masks", "_gt", "_label", "_labels",
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

    indexed: Dict[str, Path] = {}
    for path in files:
        key = sample_key(path)
        if key in indexed:
            raise RuntimeError(
                f"Duplicate sample key '{key}': {indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


def pairs_from_list(
    image_dir: str | Path,
    mask_dir: str | Path,
    list_path: str | Path,
) -> List[Tuple[Path, Path]]:
    """Resolve image/mask pairs in exactly the order listed by test.txt."""
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Split txt not found: {list_path}")

    images = index_files(image_dir)
    masks = index_files(mask_dir)
    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    seen: set[str] = set()

    with list_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first_field = line.replace(",", " ").split()[0]
            key = sample_key(Path(first_field))
            if key in seen:
                continue
            seen.add(key)
            ip, mp = images.get(key), masks.get(key)
            if ip is None or mp is None:
                missing.append(first_field)
                continue
            pairs.append((ip, mp))

    if missing:
        raise RuntimeError(
            f"{list_path} contains {len(missing)} samples that could not be paired. "
            f"First missing entries: {missing[:10]}"
        )
    if not pairs:
        raise RuntimeError(f"No pairs resolved from {list_path}")
    return pairs


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


def clean_state_dict(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in (
            "best_fixed_road_iou.pt",
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


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    weights: str,
    channels_last: bool,
) -> Tuple[nn.Module, dict, Path]:
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a train_fixed.py dictionary checkpoint")

    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise KeyError(
            "Checkpoint has no 'args'. This test script expects a checkpoint "
            "saved by train_fixed.py."
        )

    # Rebuild exactly the training architecture, but never download ImageNet
    # weights because the full road checkpoint will replace every parameter.
    model_args = dict(saved_args)
    model_args["imagenet_pretrained"] = False
    model_args["encoder_weights_path"] = None
    model = build_model(argparse.Namespace(**model_args))

    state = checkpoint.get(weights)
    if not isinstance(state, dict):
        fallback = "model" if weights == "ema" else "ema"
        state = checkpoint.get(fallback, checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise KeyError(f"No usable '{weights}', model, ema, or state_dict weights found")

    model.load_state_dict(clean_state_dict(state), strict=True)
    model = model.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model, checkpoint, checkpoint_path


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


def amp_settings(mode: str, device: torch.device) -> Tuple[bool, torch.dtype]:
    if device.type != "cuda" or mode == "none":
        return False, torch.float32
    if mode == "bfloat16":
        return True, torch.bfloat16
    if mode == "auto" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    # float16, and auto on devices without native bf16 support (for example T4).
    return True, torch.float16


def main_logits(output) -> Tensor:
    # train_fixed.py uses the final tuple element as the segmentation logits.
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("Dictionary model output has no 'logits' entry")
        output = output["logits"]
    if isinstance(output, tuple):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output


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
    """Return trainer-matched Hann-blended logits [1,2,H,W]."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Expected x with shape [1, C, H, W]")
    if stride < 1 or stride > window:
        raise ValueError("stride must satisfy 1 <= stride <= window")

    device = next(model.parameters()).device
    original_h, original_w = x.shape[-2:]
    pad_h = max(0, window - original_h)
    pad_w = max(0, window - original_w)
    if pad_h or pad_w:
        mode = "reflect" if min(original_h, original_w) > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    height, width = x.shape[-2:]
    ys = sliding_positions(height, window, stride)
    xs = sliding_positions(width, window, stride)
    coordinates = [(y, xx) for y in ys for xx in xs]

    accumulator = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
    normalizer = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    weight = hann_weight(window, device)
    amp_enabled, amp_dtype = amp_settings(amp, device)

    for start in range(0, len(coordinates), tile_batch_size):
        batch_coords = coordinates[start : start + tile_batch_size]
        tiles = torch.cat(
            [x[:, :, y : y + window, xx : xx + window] for y, xx in batch_coords],
            dim=0,
        ).to(device, non_blocking=True)
        if channels_last:
            tiles = tiles.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = main_logits(model(tiles))
        logits = logits.float()

        for index, (y, xx) in enumerate(batch_coords):
            accumulator[:, :, y : y + window, xx : xx + window] += (
                logits[index : index + 1] * weight
            )
            normalizer[:, :, y : y + window, xx : xx + window] += weight

    blended = accumulator / normalizer.clamp_min_(1e-6)
    return blended[:, :, :original_h, :original_w]


@torch.inference_mode()
def sliding_probabilities_uniform(
    model: nn.Module,
    x: Tensor,
    window: int,
    stride: int,
    tile_batch_size: int,
    amp: str,
    channels_last: bool,
) -> Tensor:
    """RoadX-style uniform blending of per-tile road probabilities."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Expected x with shape [1, C, H, W]")
    if stride < 1 or stride > window:
        raise ValueError("stride must satisfy 1 <= stride <= window")

    device = next(model.parameters()).device
    original_h, original_w = x.shape[-2:]
    pad_h = max(0, window - original_h)
    pad_w = max(0, window - original_w)
    if pad_h or pad_w:
        mode = "reflect" if min(original_h, original_w) > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    height, width = x.shape[-2:]
    ys = sliding_positions(height, window, stride)
    xs = sliding_positions(width, window, stride)
    coordinates = [(y, xx) for y in ys for xx in xs]
    accumulator = torch.zeros(
        (1, 1, height, width), device=device, dtype=torch.float32
    )
    normalizer = torch.zeros_like(accumulator)
    amp_enabled, amp_dtype = amp_settings(amp, device)

    for start in range(0, len(coordinates), tile_batch_size):
        batch_coords = coordinates[start : start + tile_batch_size]
        tiles = torch.cat(
            [x[:, :, y : y + window, xx : xx + window] for y, xx in batch_coords],
            dim=0,
        ).to(device, non_blocking=True)
        if channels_last:
            tiles = tiles.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            probabilities = road_probability(main_logits(model(tiles)))
        probabilities = probabilities.float()

        for index, (y, xx) in enumerate(batch_coords):
            accumulator[:, :, y : y + window, xx : xx + window] += (
                probabilities[index : index + 1]
            )
            normalizer[:, :, y : y + window, xx : xx + window] += 1.0

    blended = accumulator / normalizer.clamp_min_(1.0)
    return blended[:, :, :original_h, :original_w]


def tta_tags(mode: str) -> Tuple[str, ...]:
    if mode == "none":
        return ("r0",)
    if mode == "roadx3":
        # Supplied roadx.infer profile: identity, horizontal, vertical.
        return ("r0", "fr0", "fr2")
    if mode == "flip4":
        # identity, horizontal, vertical, and horizontal+vertical
        return ("r0", "fr0", "fr2", "r2")
    if mode == "d4":
        return ("r0", "r1", "r2", "r3", "fr0", "fr1", "fr2", "fr3")
    raise ValueError(f"Unsupported TTA mode: {mode}")


def apply_tta(tensor: Tensor, tag: str) -> Tensor:
    flipped = tag.startswith("f")
    rotations = int(tag[-1])
    output = torch.rot90(tensor, rotations, dims=(-2, -1))
    if flipped:
        output = torch.flip(output, dims=(-1,))
    return output


def road_probability(logits: Tensor) -> Tensor:
    """Convert one- or two-class segmentation logits to [N,1,H,W]."""
    if logits.ndim != 4:
        raise ValueError(f"Expected 4-D logits, got shape {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return logits.sigmoid()
    if logits.shape[1] == 2:
        return logits.softmax(dim=1)[:, 1:2]
    raise ValueError(
        f"Expected one or two output channels, got {logits.shape[1]}"
    )


def pad_to_multiple(x: Tensor, multiple: int) -> Tuple[Tensor, Tuple[int, int]]:
    """Reflect-pad bottom/right so H and W are divisible by ``multiple``."""
    original_h, original_w = x.shape[-2:]
    pad_h = (multiple - original_h % multiple) % multiple
    pad_w = (multiple - original_w % multiple) % multiple
    if pad_h or pad_w:
        mode = "reflect" if min(original_h, original_w) > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (original_h, original_w)


def invert_tta(tensor: Tensor, tag: str) -> Tensor:
    flipped = tag.startswith("f")
    rotations = int(tag[-1])
    output = torch.flip(tensor, dims=(-1,)) if flipped else tensor
    return torch.rot90(output, -rotations, dims=(-2, -1))


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image: np.ndarray,
    window: int = 1024,
    stride: int = 512,
    tta_mode: str = "none",
    tta_merge: str = "probabilities",
    amp: str = "auto",
    tile_batch_size: int = 1,
    channels_last: bool = True,
) -> np.ndarray:
    """Run trainer-matched sliding inference with optional full-image TTA."""
    x = image_to_tensor(image)

    if tta_mode == "roadx3":
        # Match supplied RoadX padding, views, uniform tile blending, and
        # probability averaging. Invert each complete reconstructed canvas so
        # transformed tile coordinates return to the correct original region.
        x, (original_h, original_w) = pad_to_multiple(x, stride)
        total_probability: Tensor | None = None
        tags = tta_tags(tta_mode)
        for tag in tags:
            transformed = apply_tta(x, tag)
            probability = sliding_probabilities_uniform(
                model,
                transformed,
                window=window,
                stride=stride,
                tile_batch_size=tile_batch_size,
                amp=amp,
                channels_last=channels_last,
            )
            probability = invert_tta(probability, tag)
            total_probability = (
                probability
                if total_probability is None
                else total_probability + probability
            )

        assert total_probability is not None
        mean_probability = total_probability / float(len(tags))
        return mean_probability[0, 0, :original_h, :original_w].cpu().numpy()

    total_logits: Tensor | None = None
    total_probability: Tensor | None = None
    tags = tta_tags(tta_mode)
    for tag in tags:
        x_aug = apply_tta(x, tag)
        logits = sliding_logits(
            model,
            x_aug,
            window=window,
            stride=stride,
            tile_batch_size=tile_batch_size,
            amp=amp,
            channels_last=channels_last,
        )
        logits = invert_tta(logits, tag)
        if tta_merge == "probabilities":
            probability = road_probability(logits)
            total_probability = (
                probability
                if total_probability is None
                else total_probability + probability
            )
        elif tta_merge == "logits":
            total_logits = logits if total_logits is None else total_logits + logits
        else:
            raise ValueError(f"Unsupported TTA merge mode: {tta_merge}")

    if tta_merge == "probabilities":
        assert total_probability is not None
        mean_probability = total_probability / float(len(tags))
        return mean_probability[0, 0].cpu().numpy()

    assert total_logits is not None
    mean_logits = total_logits / float(len(tags))
    return road_probability(mean_logits)[0, 0].cpu().numpy()


def counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    tn = int(np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "background_iou": background_iou,
        "miou": 0.5 * (iou + background_iou),
        "accuracy": accuracy,
    }


def relaxed_components(
    pred: np.ndarray,
    gt: np.ndarray,
    buffer_px: int,
) -> Tuple[int, int, int, int]:
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    kernel = 2 * buffer_px + 1
    pred_dilated = F.max_pool2d(
        pred_t, kernel, stride=1, padding=buffer_px
    ) > 0
    gt_dilated = F.max_pool2d(
        gt_t, kernel, stride=1, padding=buffer_px
    ) > 0
    pred_b = pred_t.bool()
    gt_b = gt_t.bool()
    return (
        int((pred_b & gt_dilated).sum()),
        int((gt_b & pred_dilated).sum()),
        int(pred_b.sum()),
        int(gt_b.sum()),
    )


def score_maps(
    probabilities: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    threshold: float,
    relaxed_buffer_px: int,
) -> Tuple[Dict[str, float], float, float, float]:
    if len(probabilities) != len(ground_truths):
        raise ValueError("probabilities and ground_truths must have equal length")
    if not probabilities:
        raise ValueError("No predictions to score")

    pooled = [0, 0, 0, 0]
    per_image_f1: List[float] = []
    per_image_iou: List[float] = []
    relaxed = [0, 0, 0, 0]

    for probability, gt in zip(probabilities, ground_truths):
        pred = probability >= threshold
        tp, fp, fn, tn = counts(pred, gt)
        for i, value in enumerate((tp, fp, fn, tn)):
            pooled[i] += value
        m = metrics_from_counts(tp, fp, fn, tn)
        per_image_f1.append(m["f1"])
        per_image_iou.append(m["iou"])
        components = relaxed_components(pred, gt, relaxed_buffer_px)
        for i, value in enumerate(components):
            relaxed[i] += value

    pooled_metrics = metrics_from_counts(*pooled)
    relaxed_precision = relaxed[0] / max(relaxed[2], 1)
    relaxed_recall = relaxed[1] / max(relaxed[3], 1)
    relaxed_f1 = (
        2.0 * relaxed_precision * relaxed_recall
        / max(relaxed_precision + relaxed_recall, 1e-12)
    )
    return (
        pooled_metrics,
        float(np.mean(per_image_f1)),
        float(np.mean(per_image_iou)),
        float(relaxed_f1),
    )


def pooled_metrics_at_threshold(
    probabilities: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    pooled = [0, 0, 0, 0]
    for probability, gt in zip(probabilities, ground_truths):
        values = counts(probability >= threshold, gt)
        for index, value in enumerate(values):
            pooled[index] += value
    return metrics_from_counts(*pooled)


def load_cache(path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    data = np.load(path, allow_pickle=True)
    probs = [np.asarray(x, dtype=np.float32) for x in data["probs"]]
    gts = [np.asarray(x, dtype=np.uint8) for x in data["gts"]]
    names = [str(x) for x in data["names"]] if "names" in data else []
    return probs, gts, names


def save_cache(
    path: Path,
    probabilities: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Native Massachusetts images can differ in shape, so object arrays are
    # intentional here. load_cache uses allow_pickle=True.
    prob_objects = np.empty(len(probabilities), dtype=object)
    gt_objects = np.empty(len(ground_truths), dtype=object)
    for index, probability in enumerate(probabilities):
        # Keep float32 so reloading a cache cannot move pixels across the
        # fixed/calibrated threshold and change the reported IoU.
        prob_objects[index] = probability.astype(np.float32)
    for index, gt in enumerate(ground_truths):
        gt_objects[index] = gt.astype(np.uint8)
    np.savez_compressed(
        path,
        probs=prob_objects,
        gts=gt_objects,
        names=np.asarray(list(names), dtype=object),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Native-resolution DualBranchRoadNet evaluation",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Exact checkpoint file is recommended",
    )
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument(
        "--tta-mode",
        choices=("none", "roadx3", "flip4", "d4"),
        default="none",
        help=(
            "none matches trainer validation; roadx3 uses corrected 3-view "
            "RoadX TTA; flip4/d4 retain trainer-matched sliding inference"
        ),
    )
    ap.add_argument(
        "--tta-merge",
        choices=("probabilities", "logits"),
        default="probabilities",
        help=(
            "how to average trainer-style TTA views; probability averaging is "
            "the recommended model-averaging default"
        ),
    )
    ap.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy alias: --tta selects flip4 and --no-tta selects none",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=None,
        help="auto-read checkpoint args.val_tile_size when omitted",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="auto-compute window - checkpoint args.val_overlap when omitted",
    )
    ap.add_argument("--tile-batch-size", type=int, default=1)
    ap.add_argument(
        "--amp",
        choices=("auto", "float16", "bfloat16", "none"),
        default="auto",
    )
    ap.add_argument(
        "--subset",
        choices=("val61", "test117", "all178", "custom"),
        default="test117",
        help="Safe Massachusetts split; custom uses the supplied list unchanged",
    )
    ap.add_argument("--val-count", type=int, default=61)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit applied after subset selection",
    )
    ap.add_argument(
        "--search-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Search pooled-IoU threshold; permitted only on val61",
    )
    ap.add_argument("--threshold-min", type=float, default=0.20)
    ap.add_argument("--threshold-max", type=float, default=0.80)
    ap.add_argument("--threshold-step", type=float, default=0.02)
    ap.add_argument("--relaxed-buffer-px", type=int, default=3)
    ap.add_argument("--out", default=None, help="Optional .npz probability/GT cache")

    root = "/kaggle/input/datasets/datnguyentien204/massachu/massachusets"
    ap.add_argument("--data-root", default=root)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--test-list", default=None)
    ap.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.tta is not None:
        args.tta_mode = "flip4" if args.tta else "none"
    if not 0.0 <= args.thr <= 1.0:
        raise ValueError("--thr must be in [0, 1]")
    if args.window is not None and args.window < 32:
        raise ValueError("--window must be >= 32")
    if args.stride is not None and args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if (
        args.window is not None
        and args.stride is not None
        and args.stride > args.window
    ):
        raise ValueError("--stride must satisfy 1 <= stride <= window")
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.val_count < 1:
        raise ValueError("--val-count must be positive")
    if args.relaxed_buffer_px < 0:
        raise ValueError("--relaxed-buffer-px cannot be negative")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive")
    if not 0.0 <= args.threshold_min <= args.threshold_max <= 1.0:
        raise ValueError("Threshold search range must be inside [0, 1]")
    if args.search_threshold and args.subset != "val61":
        raise ValueError(
            "Threshold search is allowed only on --subset val61; "
            "reuse the selected threshold on test117"
        )

    root = Path(args.data_root)
    image_dir = Path(args.image_dir) if args.image_dir else root / "images"
    mask_dir = Path(args.mask_dir) if args.mask_dir else root / "labels"
    test_list = Path(args.test_list) if args.test_list else root / "test.txt"

    all_pairs = pairs_from_list(image_dir, mask_dir, test_list)
    if args.subset == "val61":
        pairs = all_pairs[: args.val_count]
    elif args.subset == "test117":
        pairs = all_pairs[args.val_count :]
    else:
        pairs = all_pairs
    if not pairs:
        raise RuntimeError(f"Subset {args.subset} contains no images")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    expected_names = [image_path.stem for image_path, _ in pairs]

    cache = Path(args.out) if args.out else None
    if cache is not None and cache.exists():
        print(f"Loading cached native probabilities: {cache}")
        probabilities, ground_truths, names = load_cache(cache)
        if len(probabilities) != len(pairs) or len(ground_truths) != len(pairs):
            raise RuntimeError(
                f"Cache has {len(probabilities)} predictions but subset "
                f"{args.subset} requires {len(pairs)}; use a new --out path"
            )
        if names and names != expected_names:
            raise RuntimeError(
                "Cache image order does not match the selected subset; "
                "use a new --out path"
            )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, checkpoint, ckpt_path = load_model(
            args.ckpt,
            device=device,
            weights=args.weights,
            channels_last=args.channels_last,
        )
        checkpoint_args = checkpoint["args"]
        saved_window = int(checkpoint_args.get("val_tile_size", 1024))
        saved_overlap = int(checkpoint_args.get("val_overlap", 256))
        window = saved_window if args.window is None else int(args.window)
        stride = (
            window - saved_overlap
            if args.stride is None
            else int(args.stride)
        )
        if window < 32:
            raise ValueError("Resolved inference window must be >= 32")
        if stride < 1 or stride > window:
            raise ValueError(
                "Resolved stride must satisfy 1 <= stride <= window; "
                f"checkpoint val_tile_size={saved_window}, "
                f"val_overlap={saved_overlap}, resolved window={window}, "
                f"stride={stride}"
            )
        epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"Checkpoint : {ckpt_path}")
        print(f"Weights    : {args.weights}")
        print(f"Epoch      : {epoch if epoch > 0 else 'unknown'}")
        print(f"Device     : {device}")
        print(f"Test list  : {test_list}")
        print(f"Subset     : {args.subset} (val_count={args.val_count})")
        print(f"Images     : {len(pairs)}")
        print(
            f"Train val  : tile={saved_window} | overlap={saved_overlap} | "
            f"stride={saved_window - saved_overlap}"
        )
        if args.tta_mode == "roadx3":
            inference_profile = (
                "stride-multiple reflect pad | uniform PROB blending | "
                "3-view corrected RoadX TTA"
            )
        else:
            inference_profile = "Hann LOGIT blending"
        print(
            f"Inference  : native resolution | window={window} | "
            f"stride={stride} | overlap={window - stride} | "
            f"{inference_profile} | TTA={args.tta_mode} | "
            f"merge={args.tta_merge} | AMP={args.amp}"
        )

        probabilities: List[np.ndarray] = []
        ground_truths: List[np.ndarray] = []
        names: List[str] = []

        for index, (image_path, mask_path) in enumerate(pairs, start=1):
            image = read_rgb(image_path)
            gt = read_binary_mask(mask_path)
            if image.shape[:2] != gt.shape:
                raise RuntimeError(f"Shape mismatch: {image_path} vs {mask_path}")

            probability = predict_image(
                model,
                image,
                window=window,
                stride=stride,
                tta_mode=args.tta_mode,
                tta_merge=args.tta_merge,
                amp=args.amp,
                tile_batch_size=args.tile_batch_size,
                channels_last=args.channels_last,
            )

            h = min(probability.shape[0], gt.shape[0])
            w = min(probability.shape[1], gt.shape[1])
            probabilities.append(probability[:h, :w])
            ground_truths.append(gt[:h, :w])
            names.append(image_path.stem)

            pred = probability[:h, :w] >= args.thr
            tp, fp, fn, tn = counts(pred, gt[:h, :w])
            m = metrics_from_counts(tp, fp, fn, tn)
            print(
                f"[{index:3d}/{len(pairs)}] {image_path.name:<28} "
                f"F1={m['f1']:.4f} IoU={m['iou']:.4f}",
                flush=True,
            )

        if cache is not None:
            save_cache(cache, probabilities, ground_truths, names)
            print(f"Saved cache: {cache}")

    pooled, mean_f1, mean_iou, relaxed_f1 = score_maps(
        probabilities,
        ground_truths,
        args.thr,
        args.relaxed_buffer_px,
    )
    print("=" * 72)
    print(f"THRESHOLD {args.thr:.3f}")
    print(
        f"POOLED   P={pooled['precision']:.4f} "
        f"R={pooled['recall']:.4f} "
        f"F1={pooled['f1']:.4f} "
        f"IoU={pooled['iou']:.4f} "
        f"BG-IoU={pooled['background_iou']:.4f} "
        f"mIoU={pooled['miou']:.4f} "
        f"Acc={pooled['accuracy']:.4f}"
    )
    print(f"MEAN-IMG F1={mean_f1:.4f} IoU={mean_iou:.4f}")
    print(
        f"RELAXED ±{args.relaxed_buffer_px}px F1={relaxed_f1:.4f}"
    )

    if args.search_threshold:
        best_threshold = args.thr
        best_iou = -1.0
        threshold = args.threshold_min
        while threshold <= args.threshold_max + 1e-9:
            candidate = pooled_metrics_at_threshold(
                probabilities, ground_truths, threshold
            )
            if candidate["iou"] > best_iou:
                best_iou = candidate["iou"]
                best_threshold = threshold
            threshold += args.threshold_step
        print(
            f"VAL-CALIBRATED threshold={best_threshold:.2f} "
            f"pooled road IoU={best_iou:.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
