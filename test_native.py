"""Trainer-matched Massachusetts/DeepGlobe evaluation with TTA and WeavingUnet-compatible metrics.

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

Massachusetts subsets come from test.txt. DeepGlobe uses split_manifest.json
when available; otherwise it deterministically regenerates the exact training
split, including val300 inside the full test1226 set.
"""
from __future__ import annotations

import argparse
import json
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


def index_files(
    folder: str | Path,
    role: str | None = None,
) -> Dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if role is not None:
        if role not in {"image", "mask"}:
            raise ValueError("role must be image, mask, or None")
        files = [
            path
            for path in files
            if any(path.stem.lower().endswith(s) for s in MASK_SUFFIXES)
            == (role == "mask")
        ]
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


def build_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> List[Tuple[Path, Path]]:
    """Pair files by stem, including DeepGlobe's shared train directory."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    same_folder = image_dir.resolve() == mask_dir.resolve()
    images = index_files(image_dir, role="image" if same_folder else None)
    masks = index_files(mask_dir, role="mask" if same_folder else None)
    common = sorted(images.keys() & masks.keys())
    if len(common) != len(images) or len(common) != len(masks):
        raise RuntimeError(
            "Image/mask pairing mismatch: "
            f"images={len(images)}, masks={len(masks)}, pairs={len(common)}"
        )
    return [(images[key], masks[key]) for key in common]


def regenerate_deepglobe_split(
    image_dir: str | Path,
    mask_dir: str | Path,
    train_count: int,
    val_from_test_count: int,
    split_seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Reproduce train.py's deterministic overlapping DeepGlobe split."""
    all_pairs = build_pairs(image_dir, mask_dir)
    total = len(all_pairs)
    if not 0 < train_count < total:
        raise ValueError(
            f"train_count={train_count} is invalid for {total} labeled pairs"
        )
    generator = np.random.default_rng(split_seed)
    indices = generator.permutation(total)
    test_pairs = [all_pairs[int(i)] for i in indices[train_count:]]
    if not 0 < val_from_test_count <= len(test_pairs):
        raise ValueError(
            f"val_from_test_count={val_from_test_count} is invalid for "
            f"{len(test_pairs)} test pairs"
        )
    return test_pairs[:val_from_test_count], test_pairs


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


def pairs_from_manifest(
    manifest_path: str | Path,
    split: str,
) -> Tuple[List[Tuple[Path, Path]], dict]:
    """Load an exact train.py split without regenerating random indices."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Invalid split manifest: {manifest_path}")
    raw_pairs = manifest.get(split)
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise RuntimeError(
            f"Manifest split '{split}' is absent or empty: {manifest_path}"
        )
    pairs: List[Tuple[Path, Path]] = []
    for index, item in enumerate(raw_pairs):
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError(
                f"Invalid {split}[{index}] entry in {manifest_path}: {item!r}"
            )
        image_path, mask_path = Path(item[0]), Path(item[1])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Manifest pair does not exist: {image_path} | {mask_path}"
            )
        pairs.append((image_path, mask_path))
    return pairs, manifest


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
    deploy: bool = False,
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

    # IMPORTANT: load the TRAINING-form checkpoint first.  Only after all
    # RepVGG/RepDepthwise branches and BN statistics are restored do we fuse
    # them into their deploy convolutions.  Building deploy=True before loading
    # would change the state-dict keys and make the training checkpoint invalid.
    model.load_state_dict(clean_state_dict(state), strict=True)
    model = model.to(device).eval()

    if deploy:
        if not hasattr(model, "switch_to_deploy"):
            raise AttributeError(
                "This model has no switch_to_deploy() method; cannot enable deploy mode"
            )
        model.switch_to_deploy()
        model.eval()

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


_cudnn_fallback_active = False


def _forward_with_cudnn_fallback(model: nn.Module, tiles: Tensor):
    """Run one forward pass, retrying with cuDNN disabled if it can't pick a kernel.

    Grouped/depthwise convolutions (RepDepthwiseBlock's branch_3x3) under fp16
    autocast can raise "RuntimeError: GET was unable to find an engine to
    execute this computation" on some cuDNN/driver combinations, even though
    the identical model trains fine under the same AMP settings -- this is a
    known cuDNN v8 heuristic gap for certain grouped-conv shapes, not a bug in
    the model. Falling back to the slower non-cuDNN convolution keeps
    inference correct instead of crashing. Once the fallback is needed, it
    stays on for the rest of the process so later tiles don't pay for a
    repeated failed attempt.
    """
    global _cudnn_fallback_active
    if _cudnn_fallback_active:
        with torch.backends.cudnn.flags(enabled=False):
            return model(tiles)
    try:
        return model(tiles)
    except RuntimeError as error:
        if "unable to find an engine" not in str(error):
            raise
        _cudnn_fallback_active = True
        with torch.backends.cudnn.flags(enabled=False):
            return model(tiles)


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
            logits = main_logits(_forward_with_cudnn_fallback(model, tiles))
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
            probabilities = road_probability(
                main_logits(_forward_with_cudnn_fallback(model, tiles))
            )
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
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """Score predictions using both pooled and WeavingUnet-style aggregation.

    POOLED metrics are computed from one global confusion matrix over all pixels.

    WEAVING-STYLE follows the public WeavingUnet evaluation code for BOTH
    Massachusetts and DeepGlobe (their eval scripts use the same aggregation):
      * Precision  = mean(per-image precision)
      * Recall     = mean(per-image recall)
      * F1         = mean(per-image F1)
      * Accuracy   = mean(per-image accuracy)
      * IoU        = global/pooled road IoU

    Mean-image IoU and relaxed F1 are retained as additional diagnostics.
    """
    if len(probabilities) != len(ground_truths):
        raise ValueError("probabilities and ground_truths must have equal length")
    if not probabilities:
        raise ValueError("No predictions to score")

    pooled = [0, 0, 0, 0]
    per_image_precision: List[float] = []
    per_image_recall: List[float] = []
    per_image_f1: List[float] = []
    per_image_iou: List[float] = []
    per_image_accuracy: List[float] = []
    relaxed = [0, 0, 0, 0]

    for probability, gt in zip(probabilities, ground_truths):
        pred = probability >= threshold
        tp, fp, fn, tn = counts(pred, gt)

        # Global confusion counts used for pooled metrics and WeavingUnet road IoU.
        for i, value in enumerate((tp, fp, fn, tn)):
            pooled[i] += value

        # Per-image metrics, averaged later to reproduce WeavingUnet's P/R/F1/Acc.
        m = metrics_from_counts(tp, fp, fn, tn)
        per_image_precision.append(m["precision"])
        per_image_recall.append(m["recall"])
        per_image_f1.append(m["f1"])
        per_image_iou.append(m["iou"])
        per_image_accuracy.append(m["accuracy"])

        components = relaxed_components(pred, gt, relaxed_buffer_px)
        for i, value in enumerate(components):
            relaxed[i] += value

    pooled_metrics = metrics_from_counts(*pooled)

    weaving_metrics = {
        "precision": float(np.mean(per_image_precision)),
        "recall": float(np.mean(per_image_recall)),
        "f1": float(np.mean(per_image_f1)),
        # Their IOUMetric first accumulates the dataset confusion histogram,
        # therefore road IoU corresponds to our pooled/global road IoU.
        "iou": float(pooled_metrics["iou"]),
        "accuracy": float(np.mean(per_image_accuracy)),
    }

    relaxed_precision = relaxed[0] / max(relaxed[2], 1)
    relaxed_recall = relaxed[1] / max(relaxed[3], 1)
    relaxed_f1 = (
        2.0 * relaxed_precision * relaxed_recall
        / max(relaxed_precision + relaxed_recall, 1e-12)
    )

    return (
        pooled_metrics,
        weaving_metrics,
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
    ap.add_argument(
        "--dataset",
        choices=("massachusetts", "deepglobe"),
        default=None,
        help="Auto-read checkpoint args.dataset when omitted",
    )
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument(
        "--deploy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse RepVGGBlock/RepDepthwiseBlock training branches into deploy "
            "convolutions after loading the checkpoint"
        ),
    )
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
        choices=(
            "val61", "test117", "all178", "custom",
            "deepglobe_val300", "deepglobe_test1226",
        ),
        default=None,
        help=(
            "Default is test117 for Massachusetts and deepglobe_test1226 for "
            "DeepGlobe"
        ),
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

    ap.add_argument(
        "--data-root",
        default=None,
        help="Auto-select the known Massachusetts/DeepGlobe Kaggle root",
    )
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--test-list", default=None)
    ap.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "DeepGlobe split_manifest.json; defaults to the checkpoint "
            "directory"
        ),
    )
    ap.add_argument("--split-seed", type=int, default=3407)
    ap.add_argument("--deepglobe-train-count", type=int, default=5000)
    ap.add_argument("--deepglobe-val-from-test-count", type=int, default=300)
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

    # Resolve the dataset before selecting pairs. This small metadata load also
    # lets a DeepGlobe checkpoint find the manifest beside itself by default.
    resolved_ckpt = resolve_checkpoint(args.ckpt)
    if args.dataset is None:
        try:
            metadata = torch.load(
                resolved_ckpt, map_location="cpu", weights_only=False
            )
        except TypeError:
            metadata = torch.load(resolved_ckpt, map_location="cpu")
        saved_args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
        args.dataset = str(saved_args.get("dataset", "massachusetts"))
        del metadata
    if args.dataset not in {"massachusetts", "deepglobe"}:
        raise ValueError(f"Unsupported checkpoint dataset: {args.dataset}")
    if args.subset is None:
        args.subset = (
            "deepglobe_test1226"
            if args.dataset == "deepglobe"
            else "test117"
        )
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
    if args.deepglobe_train_count < 1:
        raise ValueError("--deepglobe-train-count must be positive")
    if args.deepglobe_val_from_test_count < 1:
        raise ValueError("--deepglobe-val-from-test-count must be positive")
    if args.relaxed_buffer_px < 0:
        raise ValueError("--relaxed-buffer-px cannot be negative")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive")
    if not 0.0 <= args.threshold_min <= args.threshold_max <= 1.0:
        raise ValueError("Threshold search range must be inside [0, 1]")
    calibration_subsets = {"val61", "deepglobe_val300"}
    if args.search_threshold and args.subset not in calibration_subsets:
        raise ValueError(
            "Threshold search is allowed only on val61 or deepglobe_val300; "
            "reuse the selected threshold on the corresponding full test set"
        )

    if args.dataset == "deepglobe":
        if args.subset not in {"deepglobe_val300", "deepglobe_test1226"}:
            raise ValueError(
                "DeepGlobe requires --subset deepglobe_val300 or "
                "deepglobe_test1226"
            )
        manifest_path = (
            Path(args.split_manifest)
            if args.split_manifest
            else resolved_ckpt.parent / "split_manifest.json"
        )
        manifest_split = (
            "val" if args.subset == "deepglobe_val300" else "test"
        )
        if manifest_path.is_file():
            pairs, manifest = pairs_from_manifest(manifest_path, manifest_split)
            overlap_counts = manifest.get("overlap_counts", {})
            if int(overlap_counts.get("val_test", -1)) != 300:
                raise RuntimeError(
                    "Manifest does not record the requested 300-image val/test "
                    f"overlap: {overlap_counts}"
                )
            split_source = manifest_path
        else:
            deepglobe_root = Path(
                args.data_root
                or "/kaggle/input/datasets/balraj98/"
                "deepglobe-road-extraction-dataset"
            )
            image_dir = (
                Path(args.image_dir)
                if args.image_dir
                else deepglobe_root / "train"
            )
            mask_dir = (
                Path(args.mask_dir)
                if args.mask_dir
                else deepglobe_root / "train"
            )
            val_pairs, test_pairs = regenerate_deepglobe_split(
                image_dir=image_dir,
                mask_dir=mask_dir,
                train_count=args.deepglobe_train_count,
                val_from_test_count=args.deepglobe_val_from_test_count,
                split_seed=args.split_seed,
            )
            pairs = val_pairs if manifest_split == "val" else test_pairs
            split_source = Path(
                "regenerated:"
                f"seed={args.split_seed},train={args.deepglobe_train_count},"
                f"val_from_test={args.deepglobe_val_from_test_count}"
            )
        expected_count = 300 if manifest_split == "val" else 1226
        if len(pairs) != expected_count:
            raise RuntimeError(
                f"{args.subset} requires {expected_count} pairs, but the "
                f"resolved split contains {len(pairs)}"
            )
    else:
        if args.subset not in {"val61", "test117", "all178", "custom"}:
            raise ValueError(
                "Massachusetts requires val61, test117, all178, or custom"
            )
        root = Path(
            args.data_root
            or "/kaggle/input/datasets/datnguyentien204/massachu/massachusets"
        )
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
        split_source = test_list
    if not pairs:
        raise RuntimeError(f"Subset {args.subset} contains no images")
    if args.limit is not None:
        pairs = pairs[: args.limit]
    expected_names = [image_path.stem for image_path, _ in pairs]

    cache = Path(args.out) if args.out else None
    if cache is not None and args.deploy:
        # Never reuse a non-deploy probability cache for a deploy evaluation.
        cache = cache.with_name(f"{cache.stem}_deploy{cache.suffix}")
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
            deploy=args.deploy,
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
        print(f"Deploy     : {'ON' if args.deploy else 'OFF'}")
        print(f"Epoch      : {epoch if epoch > 0 else 'unknown'}")
        print(f"Device     : {device}")
        print(f"Dataset    : {args.dataset}")
        print(f"Split file : {split_source}")
        if args.dataset == "deepglobe":
            print(
                f"Subset     : {args.subset} "
                "(val300 is contained in test1226)"
            )
        else:
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
            f"merge={args.tta_merge} | AMP={args.amp} | "
            f"deploy={'on' if args.deploy else 'off'}"
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

    pooled, weaving, mean_iou, relaxed_f1 = score_maps(
        probabilities,
        ground_truths,
        args.thr,
        args.relaxed_buffer_px,
    )
    print("=" * 72)
    print(f"THRESHOLD {args.thr:.3f}")
    print(
        f"METRIC PROTOCOL : WeavingUnet-compatible for {args.dataset} "
        "(mean-image P/R/F1/Acc + pooled/global road IoU)"
    )
    print(
        f"POOLED       P={pooled['precision']:.4f} "
        f"R={pooled['recall']:.4f} "
        f"F1={pooled['f1']:.4f} "
        f"IoU={pooled['iou']:.4f} "
        f"BG-IoU={pooled['background_iou']:.4f} "
        f"mIoU={pooled['miou']:.4f} "
        f"Acc={pooled['accuracy']:.4f}"
    )
    print(
        f"WEAVING-STYLE P={weaving['precision']:.4f} "
        f"R={weaving['recall']:.4f} "
        f"F1={weaving['f1']:.4f} "
        f"IoU={weaving['iou']:.4f} "
        f"Acc={weaving['accuracy']:.4f}"
    )
    print(
        f"MEAN-IMG     F1={weaving['f1']:.4f} "
        f"IoU={mean_iou:.4f}"
    )
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
