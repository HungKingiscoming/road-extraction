"""Trainer-matched Massachusetts/DeepGlobe evaluation with TTA.

The default inference path is native resolution with ImageNet normalization,
reflect padding, a 1024 window with 256 overlap (stride 768), and Hann-weighted
LOGIT blending. Optional flip4 or D4 TTA applies this complete path to each
transformed full image and inverse-transforms the complete blended map. TTA
views can be merged as probabilities (recommended) or logits.

There is no validation split: the decision threshold is fixed (--thr; default 0.72
for massachusetts, 0.66 for deepglobe) and the whole test split is evaluated.

Test images: --dataset picks massachusetts or deepglobe (default: the one the
checkpoint was trained on) and the images are read from data/<dataset>/test
(or <--data-root>/test); images and labels are paired by file name.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modeling.model import build_model


DEFAULT_THRESHOLDS = {"massachusetts": 0.72, "deepglobe": 0.66}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


# Dataset folders: data/<dataset>/{train,test} (or training/eval). Inside a
# folder, files are classified as image or label and paired by file name.
DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
MASK_DIR_WORDS = {
    "label", "labels", "mask", "masks", "gt", "groundtruth",
    "annotation", "annotations",
}
SPLIT_DIRS = {"train": ("train", "training"), "test": ("test", "eval")}

Pair = Tuple[Path, Path]


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in (
        "_image", "_images", "_img", "_sat",
        "_mask", "_masks", "_gt", "_label", "_labels",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def is_mask_file(path: Path, folder: Path) -> bool:
    if any(path.stem.lower().endswith(s) for s in MASK_SUFFIXES):
        return True
    for part in path.relative_to(folder).parts[:-1]:
        if MASK_DIR_WORDS & set(re.split(r"[^a-z0-9]+", part.lower())):
            return True
    return False


def collect_pairs(folder: Path) -> List[Pair]:
    """Find every image and label under ``folder`` and pair them by name."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}. Create it and put the "
            "images and their labels inside (see README.md)."
        )
    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = masks if is_mask_file(path, folder) else images
        target.setdefault(sample_key(path), path)
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise RuntimeError(
            f"No image/label pairs found in {folder} ({len(images)} images, "
            f"{len(masks)} labels). Put the images and their labels in that "
            "folder (see README.md)."
        )
    unmatched = (images.keys() | masks.keys()) - set(common)
    if unmatched:
        print(
            f"[data] {folder}: {len(common)} image/label pairs, ignored "
            f"{len(unmatched)} files without a partner "
            f"(e.g. {sorted(unmatched)[:3]})"
        )
    return [(images[key], masks[key]) for key in common]


def load_split(
    dataset: str, split: str, data_root: Optional[str | Path] = None
) -> List[Pair]:
    """Image/label pairs of ``split`` ("train" or "test") for ``dataset``."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"split must be one of {tuple(SPLIT_DIRS)}, got {split!r}")
    root = Path(data_root).expanduser() if data_root else DATA_DIR / dataset
    for name in SPLIT_DIRS[split]:
        if (root / name).is_dir():
            return collect_pairs(root / name)
    raise FileNotFoundError(
        f"No {split} folder in {root}: create {root / SPLIT_DIRS[split][0]} "
        f"(or {root / SPLIT_DIRS[split][1]}) and put the images and their "
        "labels inside (see README.md)."
    )


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
        candidate = path / "last.pt"
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


def score_maps(
    probabilities: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    """Precision, recall, F1 and accuracy averaged over images, plus road IoU.

    IoU comes from one confusion matrix accumulated over the whole test split.
    """
    if len(probabilities) != len(ground_truths):
        raise ValueError("probabilities and ground_truths must have equal length")
    if not probabilities:
        raise ValueError("No predictions to score")

    pooled = [0, 0, 0, 0]
    per_image = {"precision": [], "recall": [], "f1": [], "accuracy": []}

    for probability, gt in zip(probabilities, ground_truths):
        tp, fp, fn, tn = counts(probability >= threshold, gt)
        for i, value in enumerate((tp, fp, fn, tn)):
            pooled[i] += value
        m = metrics_from_counts(tp, fp, fn, tn)
        for name, values in per_image.items():
            values.append(m[name])

    return {
        **{name: float(np.mean(values)) for name, values in per_image.items()},
        "iou": metrics_from_counts(*pooled)["iou"],
    }


def parse_color(text: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--overlay-color must be 'R,G,B', got {text!r}")
    values = tuple(int(p) for p in parts)
    if any(not 0 <= v <= 255 for v in values):
        raise ValueError(f"--overlay-color channels must be in [0, 255], got {text!r}")
    return values  # type: ignore[return-value]


def save_prediction_outputs(
    pairs: Sequence[Tuple[Path, Path]],
    probabilities: Sequence[np.ndarray],
    threshold: float,
    save_dir: Path,
    tag: str,
    overlay_color: Tuple[int, int, int],
) -> None:
    """Dump per-image overlay + binary-mask predictions.

    Matches the existing images/<dataset>/<tag>/ convention used for the
    paper's qualitative comparison figures:
      {stem}_sat_{tag}_overlay.jpg  -- RGB image with predicted road pixels
                                        painted solid ``overlay_color``
      {stem}_sat_{tag}_pred_bin.png -- binary road mask (0/255)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for (image_path, _mask_path), probability in zip(pairs, probabilities):
        base_name = sample_key(image_path)
        image = read_rgb(image_path)
        pred = probability >= threshold

        height = min(image.shape[0], pred.shape[0])
        width = min(image.shape[1], pred.shape[1])
        pred = pred[:height, :width]

        overlay = image[:height, :width].copy()
        overlay[pred] = overlay_color
        binary_mask = (pred.astype(np.uint8) * 255)

        overlay_path = save_dir / f"{base_name}_sat_{tag}_overlay.jpg"
        bin_path = save_dir / f"{base_name}_sat_{tag}_pred_bin.png"
        Image.fromarray(overlay, mode="RGB").save(overlay_path, quality=95)
        Image.fromarray(binary_mask, mode="L").save(bin_path)


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
        # decision threshold and change the reported IoU.
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
        help="Dataset to test on, read from data/<dataset>/test; default: the checkpoint's",
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
    ap.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Decision threshold; default: 0.72 for massachusetts, 0.66 for deepglobe",
    )
    ap.add_argument(
        "--thr-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="THR",
        help=(
            "Also print the metrics for each of these thresholds (e.g. "
            "--thr-sweep 0.4 0.5 0.6 0.7). Use with --out so the images are "
            "only predicted once."
        ),
    )
    ap.add_argument(
        "--tta-mode",
        choices=("none", "roadx3", "flip4", "d4"),
        default="none",
        help=(
            "none is plain sliding-window inference; roadx3 uses corrected "
            "3-view RoadX TTA; flip4/d4 use flip/rotation TTA"
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
        default=1024,
        help="sliding-window tile size in pixels",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=768,
        help="sliding-window stride in pixels (window - overlap)",
    )
    ap.add_argument("--tile-batch-size", type=int, default=1)
    ap.add_argument(
        "--amp",
        choices=("auto", "float16", "bfloat16", "none"),
        default="auto",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit applied to the test images",
    )
    ap.add_argument("--out", default=None, help="Optional .npz probability/GT cache")
    ap.add_argument(
        "--save-preds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also dump per-image predictions as "
            "{stem}_sat_{model-tag}_overlay.jpg and "
            "{stem}_sat_{model-tag}_pred_bin.png, matching the images/ "
            "qualitative-comparison convention"
        ),
    )
    ap.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Output directory for --save-preds; defaults to "
            "images/<dataset>/<model-tag>"
        ),
    )
    ap.add_argument(
        "--model-tag",
        default="carnet",
        help="Model name embedded in saved filenames and the default --save-dir",
    )
    ap.add_argument(
        "--overlay-color",
        default="32,178,170",
        help="R,G,B (0-255 each) used to paint predicted road pixels in the overlay image",
    )

    ap.add_argument(
        "--data-root",
        default=None,
        help=(
            "Folder with a test/ sub-folder, used instead of data/<dataset>/"
        ),
    )
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
    if args.thr is None:
        args.thr = DEFAULT_THRESHOLDS[args.dataset]
    if not 0.0 <= args.thr <= 1.0:
        raise ValueError("--thr must be in [0, 1]")
    if any(not 0.0 <= thr <= 1.0 for thr in args.thr_sweep or []):
        raise ValueError("--thr-sweep values must be in [0, 1]")
    if args.window < 32:
        raise ValueError("--window must be >= 32")
    if args.stride < 1 or args.stride > args.window:
        raise ValueError("--stride must satisfy 1 <= stride <= window")
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    overlay_color = parse_color(args.overlay_color)

    pairs = load_split(args.dataset, "test", args.data_root)
    split_source = args.data_root or f"data/{args.dataset}"
    if not pairs:
        raise RuntimeError(f"No test images found for {args.dataset}")
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
                f"Cache has {len(probabilities)} predictions but the test split "
                f"has {len(pairs)} images; use a new --out path"
            )
        if names and names != expected_names:
            raise RuntimeError(
                "Cache image order does not match the test split; "
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
        window = int(args.window)
        stride = int(args.stride)
        epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"Checkpoint : {ckpt_path}")
        print(f"Weights    : {args.weights}")
        print(f"Deploy     : {'ON' if args.deploy else 'OFF'}")
        print(f"Epoch      : {epoch if epoch > 0 else 'unknown'}")
        print(f"Device     : {device}")
        print(f"Dataset    : {args.dataset}")
        print(f"Test folder: {split_source}/test")
        print(f"Images     : {len(pairs)}")
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

    if args.save_preds:
        save_dir = (
            Path(args.save_dir)
            if args.save_dir
            else Path("images") / args.dataset / args.model_tag
        )
        save_prediction_outputs(
            pairs, probabilities, args.thr, save_dir, args.model_tag, overlay_color
        )
        print(f"Saved {len(pairs)} prediction image pairs to {save_dir}")

    metrics = score_maps(probabilities, ground_truths, args.thr)
    print("=" * 72)
    print(
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"IoU={metrics['iou']:.4f} "
        f"Acc={metrics['accuracy']:.4f}"
    )
    print("=" * 72)
    for thr in args.thr_sweep or []:
        m = score_maps(probabilities, ground_truths, thr)
        print(
            f"thr={thr:.2f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"IoU={m['iou']:.4f} Acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
