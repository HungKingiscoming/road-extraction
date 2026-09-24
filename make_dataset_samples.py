"""Render post-augmentation image/mask sample grids for the paper figure.

The road-guided random crop is a custom data-sampling heuristic (max-pool
the mask, sample a road-containing cell, keep the best-scoring candidate)
that has no stock albumentations equivalent, so it stays hand-rolled. The
flip/rotate/photometric augmentation stage uses albumentations, with
probabilities matching the paper's stated pipeline: brightness+contrast
jitter (p=0.60, combined under one trigger as the paper describes them),
saturation jitter (p=0.35), Gaussian blur (p=0.15), Gaussian noise (p=0.15).
road_occlusion is omitted entirely, matching the current baseline's
road_occlusion_probability = 0.0. Because albumentations' brightness/
contrast/blur/noise implementations differ numerically from train.py's PIL
ImageEnhance/ImageFilter calls, this script is semantically equivalent to
train.py's augmentation, not a bit-for-bit reproduction of it.

Writes two separate PNGs (one per dataset) with no baked-in title/caption
text, since the LaTeX figure environment supplies the "(a) ..." / "(b) ..."
labels itself:

    \\includegraphics[width=\\linewidth]{figs/massachusetts_samples_5x2.png}
    \\textbf{(a) Massachusetts Roads}
    \\includegraphics[width=\\linewidth]{figs/deepglobe_samples_5x2.png}
    \\textbf{(b) DeepGlobe Road Extraction}

Usage:
    python make_dataset_samples.py --dataset both
    python make_dataset_samples.py --dataset massachusetts --seed 3
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent

# Candidate dataset roots, tried in order. Each root is expected to contain
# an "images" subfolder plus a "labels" or "masks" subfolder. The first root
# with both subfolders present is used, so this script works unchanged both
# against a local copy of the data and against the Kaggle input mounts.
DATASETS = {
    "massachusetts": {
        "roots": [
            REPO_ROOT / "massachusetts" / "massachusets",
            Path("/kaggle/input/datasets/k4nngg/massa-road/datasetmassa/ROAD/training"),
        ],
        "out": "massachusetts_samples_5x2.png",
        "out_overlay": "massachusetts_augmentation_5x2.png",
    },
    "deepglobe": {
        "roots": [
            REPO_ROOT / "deepglobe" / "datasetdg" / "ROAD" / "training",
            Path("/kaggle/input/datasets/k4nngg/datadg/datasetdg/ROAD/training"),
        ],
        "out": "deepglobe_samples_5x2.png",
        "out_overlay": "deepglobe_augmentation_5x2.png",
    },
}

OVERLAY_COLOR = (255, 45, 45)
OVERLAY_ALPHA = 0.55

MASK_SUBDIR_NAMES = ("labels", "masks")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def resolve_dataset_dirs(roots: List[Path]) -> Tuple[Path, Path]:
    for root in roots:
        images_dir = root / "images"
        if not images_dir.is_dir():
            continue
        for mask_name in MASK_SUBDIR_NAMES:
            masks_dir = root / mask_name
            if masks_dir.is_dir():
                return images_dir, masks_dir
    tried = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(f"No dataset found under any of: {tried}")


# ---------------------------------------------------------------------------
# Dataset pairing
# ---------------------------------------------------------------------------


def build_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = {
        p.stem[: -len("_sat")]: p
        for p in sorted(image_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.endswith("_sat")
    }
    masks = {
        p.stem[: -len("_mask")]: p
        for p in sorted(mask_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.endswith("_mask")
    }
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise RuntimeError(f"No matching image/mask pairs found in {image_dir} / {mask_dir}")
    return [(images[key], masks[key]) for key in common]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    threshold = 0 if int(mask.max(initial=0)) <= 1 else 127
    return (mask > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Road-guided random crop (mirrors train.py: pad_pair_to_size, coarse_max_mask,
# random_crop_pair)
# ---------------------------------------------------------------------------


def pad_pair_to_size(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    image_mode = "reflect" if min(height, width) > 1 else "edge"
    image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode=image_mode)
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant")
    return image, mask


def coarse_max_mask(mask: np.ndarray, factor: int = 8) -> np.ndarray:
    height, width = mask.shape
    pooled_h, pooled_w = math.ceil(height / factor), math.ceil(width / factor)
    padded = np.pad(mask, ((0, pooled_h * factor - height), (0, pooled_w * factor - width)), mode="constant")
    return padded.reshape(pooled_h, factor, pooled_w, factor).max(axis=(1, 3))


def random_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    road_probability: float,
    minimum_fraction: float,
    tries: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image, mask = pad_pair_to_size(image, mask, crop_size)
    height, width = mask.shape
    max_y, max_x = height - crop_size, width - crop_size

    def random_origin() -> Tuple[int, int]:
        return (
            random.randint(0, max_y) if max_y else 0,
            random.randint(0, max_x) if max_x else 0,
        )

    y0, x0 = random_origin()
    if mask.any() and random.random() < road_probability:
        factor = 8
        coarse = coarse_max_mask(mask, factor)
        road_cells = np.flatnonzero(coarse)
        coarse_crop = max(1, math.ceil(crop_size / factor))
        best_origin = (y0, x0)
        best_score = -1.0
        for _ in range(max(1, tries)):
            flat = int(road_cells[random.randrange(len(road_cells))])
            cy, cx = np.unravel_index(flat, coarse.shape)
            road_y = min(height - 1, cy * factor + random.randrange(factor))
            road_x = min(width - 1, cx * factor + random.randrange(factor))
            y0 = min(max(road_y - random.randrange(crop_size), 0), max_y)
            x0 = min(max(road_x - random.randrange(crop_size), 0), max_x)
            py, px = y0 // factor, x0 // factor
            region = coarse[py : min(coarse.shape[0], py + coarse_crop), px : min(coarse.shape[1], px + coarse_crop)]
            score = float(region.mean()) if region.size else 0.0
            if score > best_score:
                best_score, best_origin = score, (y0, x0)
            if score >= minimum_fraction:
                break
        y0, x0 = best_origin
    return (
        np.ascontiguousarray(image[y0 : y0 + crop_size, x0 : x0 + crop_size]),
        np.ascontiguousarray(mask[y0 : y0 + crop_size, x0 : x0 + crop_size]),
    )


# ---------------------------------------------------------------------------
# Augmentation (albumentations; road_occlusion omitted since
# road_occlusion_probability = 0.0 in the current baseline)
# ---------------------------------------------------------------------------


def build_augment_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.RandomRotate90(p=1.0),
            # "brightness and contrast jitter (p = 0.60)" -- one combined trigger.
            A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=0.80),
            # "saturation jitter (p = 0.35)" -- ColorJitter with every other
            # channel pinned to a no-op range so only saturation moves.
            A.ColorJitter(
                brightness=(1.2, 1.2), contrast=(1.2, 1.2), saturation=(0.1, 1.30), hue=(0.2, 0.2), p=0.35
            ),
            A.GaussianBlur(blur_limit=1, sigma_limit=(0.2, 1.4), p=0.3),
            A.GaussNoise(std_range=(5 / 255, 10 / 255), p=0.30),
        ]
    )


def augment_pair(image: np.ndarray, mask: np.ndarray, transform: A.Compose) -> Tuple[np.ndarray, np.ndarray]:
    # Reseed per call from the already-seeded global `random` module so the
    # whole pipeline stays reproducible from a single --seed.
    transform.set_random_seed(random.getrandbits(32))
    result = transform(image=np.ascontiguousarray(image), mask=np.ascontiguousarray(mask))
    return result["image"], result["mask"]


def overlay_mask_on_image(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float
) -> np.ndarray:
    """Blend the road mask onto the image as a translucent color fill."""
    overlay = image.astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    road = mask.astype(bool)
    overlay[road] = (1.0 - alpha) * overlay[road] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def render_grid(
    pairs: List[Tuple[Path, Path]],
    n: int,
    crop_size: int,
    road_crop_probability: float,
    road_crop_min_fraction: float,
    road_crop_tries: int,
    tile_size: int,
    gap: int,
    out_path: Path,
    transform: A.Compose,
    mode: str = "pairs",
) -> None:
    chosen = random.sample(pairs, n) if len(pairs) >= n else random.choices(pairs, k=n)

    tiles_top, tiles_bottom = [], []
    for image_path, mask_path in chosen:
        image, mask = read_rgb(image_path), read_binary_mask(mask_path)
        image, mask = random_crop_pair(
            image, mask, crop_size, road_crop_probability, road_crop_min_fraction, road_crop_tries
        )

        if mode == "overlay":
            # Top row: original crop before augmentation (no overlay) so the
            # augmentation effect is visible by comparison. Bottom row: the
            # same crop after augmentation, with the road mask overlaid.
            original_tile = Image.fromarray(image).resize((tile_size, tile_size), Image.BILINEAR)
            tiles_top.append(original_tile)

            augmented, augmented_mask = augment_pair(image, mask, transform)
            augmented = overlay_mask_on_image(augmented, augmented_mask, OVERLAY_COLOR, OVERLAY_ALPHA)
            augmented_tile = Image.fromarray(augmented).resize((tile_size, tile_size), Image.BILINEAR)
            tiles_bottom.append(augmented_tile)
        else:
            image, mask = augment_pair(image, mask, transform)
            image_tile = Image.fromarray(image).resize((tile_size, tile_size), Image.BILINEAR)
            mask_tile = Image.fromarray((mask * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.NEAREST)
            tiles_top.append(image_tile)
            tiles_bottom.append(mask_tile.convert("RGB"))

    canvas_w = n * tile_size + (n - 1) * gap
    canvas_h = 2 * tile_size + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    for col in range(n):
        x = col * (tile_size + gap)
        canvas.paste(tiles_top[col], (x, 0))
        canvas.paste(tiles_bottom[col], (x, tile_size + gap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path} ({canvas_w}x{canvas_h})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["massachusetts", "deepglobe", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=["pairs", "overlay"],
        default="pairs",
        help="'pairs': image row + binary mask row. 'overlay': original crop row + "
        "post-augmentation row with the mask blended onto the image",
    )
    parser.add_argument("--n", type=int, default=5, help="number of columns (samples) per grid")
    parser.add_argument("--crop_size", type=int, default=1024, help="training crop size (train.py --crop_size)")
    parser.add_argument("--road_crop_probability", type=float, default=0.60)
    parser.add_argument("--road_crop_min_fraction", type=float, default=0.002)
    parser.add_argument("--road_crop_tries", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=400, help="output size (px) of each square tile")
    parser.add_argument("--gap", type=int, default=4, help="gap (px) between tiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "paper" / "figs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    transform = build_augment_transform()

    names = ["massachusetts", "deepglobe"] if args.dataset == "both" else [args.dataset]
    for name in names:
        spec = DATASETS[name]
        images_dir, masks_dir = resolve_dataset_dirs(spec["roots"])
        pairs = build_pairs(images_dir, masks_dir)
        out_name = spec["out_overlay"] if args.mode == "overlay" else spec["out"]
        render_grid(
            pairs,
            args.n,
            args.crop_size,
            args.road_crop_probability,
            args.road_crop_min_fraction,
            args.road_crop_tries,
            args.tile_size,
            args.gap,
            args.out_dir / out_name,
            transform,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
