from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from modeling.decoder import RoadSegOrientationLoss, binary_dice_loss
from modeling.model import DualBranchRoadNet, build_model


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

# Optimizer parameter-group buckets, used everywhere a per-bucket metric
# (grad norm, weight norm) needs to be reported. Keep in sync with the keys
# returned by DualBranchRoadNet.optimization_modules().
OPTIMIZER_BUCKETS = ("head", "dual_branch", "layer3", "layer2", "early_encoder")


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------


def init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(
            f"[rank {rank}] Initializing NCCL on cuda:{local_rank} "
            f"(world_size={world_size})...",
            flush=True,
        )
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=15),
        )
        print(f"[rank {rank}] NCCL process group ready", flush=True)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_active() or dist.get_rank() == 0


def rank_zero_print(*values, **kwargs) -> None:
    if is_main_process():
        kwargs.setdefault("flush", True)
        print(*values, **kwargs)


def unwrap_model(model: nn.Module) -> DualBranchRoadNet:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


def cleanup_distributed() -> None:
    if distributed_active():
        dist.destroy_process_group()


class DistributedEvalSampler(Sampler[int]):
    """Shard validation exactly, without padded duplicate images."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, math.ceil(remaining / self.world_size))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Dataset discovery, native crop, and augmentation
# ---------------------------------------------------------------------------


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in (
        "_image",
        "_images",
        "_img",
        "_sat",
        "_mask",
        "_masks",
        "_gt",
        "_label",
        "_labels",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def index_files(folder: str | Path, role: Optional[str] = None) -> Dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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
        raise RuntimeError(f"No supported files found in {folder}")
    indexed: Dict[str, Path] = {}
    for path in files:
        key = sample_key(path)
        if key in indexed:
            raise RuntimeError(
                f"Duplicate sample key '{key}': {indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


def build_pairs(image_dir: str | Path, mask_dir: str | Path) -> List[Tuple[Path, Path]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    same_folder = image_dir.resolve() == mask_dir.resolve()
    images = index_files(image_dir, role="image" if same_folder else None)
    masks = index_files(mask_dir, role="mask" if same_folder else None)
    common = sorted(images.keys() & masks.keys())
    if len(common) != len(images) or len(common) != len(masks):
        raise RuntimeError(
            "Image/mask pairing mismatch: "
            f"images={len(images)}, masks={len(masks)}, pairs={len(common)}, "
            f"missing masks={sorted(images.keys() - masks.keys())[:5]}, "
            f"missing images={sorted(masks.keys() - images.keys())[:5]}"
        )
    return [(images[key], masks[key]) for key in common]


def pairs_from_list(
    image_dir: str | Path,
    mask_dir: str | Path,
    list_path: str | Path,
) -> List[Tuple[Path, Path]]:
    """Resolve pairs in exactly the order given by a txt split file."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
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
            image_path = images.get(key)
            mask_path = masks.get(key)
            if image_path is None or mask_path is None:
                missing.append(first_field)
                continue
            pairs.append((image_path, mask_path))

    if missing:
        raise RuntimeError(
            f"{list_path} contains {len(missing)} unpaired samples. "
            f"First missing entries: {missing[:10]}"
        )
    if not pairs:
        raise RuntimeError(f"No image/mask pairs resolved from {list_path}")
    return pairs


def _first_existing_pair(
    candidates: Sequence[Tuple[Path, Path]],
) -> Tuple[Path, Path]:
    for image_dir, mask_dir in candidates:
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir
    return candidates[0]


def _first_labeled_pair(
    candidates: Sequence[Tuple[Path, Path]],
) -> Optional[Tuple[Path, Path]]:
    """Return the first directory pair that contains matched masks.

    DeepGlobe mirrors commonly ship ``valid`` images without public masks.  A
    directory existing is therefore not enough to call it a validation split.
    """
    for image_dir, mask_dir in candidates:
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        try:
            if build_pairs(image_dir, mask_dir):
                return image_dir, mask_dir
        except RuntimeError:
            continue
    return None


def configure_dataset_paths(args: argparse.Namespace) -> None:
    """Resolve common Kaggle layouts while keeping explicit CLI paths final."""
    if args.dataset == "massachusetts":
        root = Path(
            args.data_root
            or "/kaggle/input/datasets/datnguyentien204/massachu/massachusets"
        )
        args.train_image_dir = args.train_image_dir or str(root / "images")
        args.train_mask_dir = args.train_mask_dir or str(root / "labels")
        args.train_list = args.train_list or str(root / "train.txt")
        args.test_list = args.test_list or str(root / "test.txt")
        args.val_image_dir = None
        args.val_mask_dir = None
        return

    root = Path(
        args.data_root
        or "/kaggle/input/datasets/balraj98/deepglobe-road-extraction-dataset"
    )
    train_images, train_masks = _first_existing_pair(
        (
            (root / "train" / "images", root / "train" / "gt"),
            (root / "train" / "images", root / "train" / "masks"),
            (root / "images", root / "gt"),
            (root / "train", root / "train"),
            (root, root),
        )
    )
    args.train_image_dir = args.train_image_dir or str(train_images)
    args.train_mask_dir = args.train_mask_dir or str(train_masks)
    if args.val_image_dir is None and args.val_mask_dir is None:
        labeled_validation = _first_labeled_pair(
            (
                (root / "valid" / "images", root / "valid" / "gt"),
                (root / "val" / "images", root / "val" / "gt"),
                (root / "valid", root / "valid"),
            )
        )
        if labeled_validation is not None:
            val_images, val_masks = labeled_validation
            args.val_image_dir = str(val_images)
            args.val_mask_dir = str(val_masks)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    threshold = 0 if int(mask.max(initial=0)) <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def pad_pair_to_size(
    image: np.ndarray, mask: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    image_mode = "reflect" if min(height, width) > 1 else "edge"
    image = np.pad(
        image, ((top, bottom), (left, right), (0, 0)), mode=image_mode
    )
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant")
    return image, mask


def coarse_max_mask(mask: np.ndarray, factor: int = 8) -> np.ndarray:
    height, width = mask.shape
    pooled_h, pooled_w = math.ceil(height / factor), math.ceil(width / factor)
    padded = np.pad(
        mask,
        ((0, pooled_h * factor - height), (0, pooled_w * factor - width)),
        mode="constant",
    )
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
            region = coarse[
                py : min(coarse.shape[0], py + coarse_crop),
                px : min(coarse.shape[1], px + coarse_crop),
            ]
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


def road_guided_occlusion(
    image: np.ndarray,
    mask: np.ndarray,
    probability: float,
    max_patches: int,
) -> np.ndarray:
    """Synthesize shadows/vegetation/vehicles over labeled road pixels."""
    if probability <= 0.0 or not mask.any() or random.random() >= probability:
        return image

    road_y, road_x = np.nonzero(mask)
    height, width = mask.shape
    pil = Image.fromarray(np.ascontiguousarray(image))
    number = random.randint(1, max(1, int(max_patches)))

    for _ in range(number):
        index = random.randrange(len(road_y))
        center_y, center_x = int(road_y[index]), int(road_x[index])
        short_side = random.randint(
            max(3, round(min(height, width) * 0.006)),
            max(5, round(min(height, width) * 0.018)),
        )
        long_side = random.randint(short_side, max(short_side + 1, short_side * 3))
        if random.random() < 0.5:
            box_width, box_height = long_side, short_side
        else:
            box_width, box_height = short_side, long_side
        x0, y0 = center_x - box_width // 2, center_y - box_height // 2
        x1, y1 = center_x + box_width // 2, center_y + box_height // 2

        alpha = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(alpha)
        if random.random() < 0.65:
            draw.ellipse((x0, y0, x1, y1), fill=random.randint(125, 210))
            alpha = alpha.filter(ImageFilter.GaussianBlur(random.uniform(1.0, 3.0)))
        else:
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=max(1, short_side // 4),
                fill=random.randint(150, 230),
            )

        mode = random.choice(("shadow", "shadow", "vegetation", "vehicle"))
        if mode == "shadow":
            factor = random.uniform(0.30, 0.65)
            overlay = ImageEnhance.Brightness(pil).enhance(factor)
        elif mode == "vegetation":
            color = (
                random.randint(25, 85),
                random.randint(65, 135),
                random.randint(20, 75),
            )
            overlay = Image.new("RGB", (width, height), color)
        else:
            value = random.randint(135, 235)
            tint = random.randint(-15, 15)
            color = (
                int(np.clip(value + tint, 0, 255)),
                value,
                int(np.clip(value - tint, 0, 255)),
            )
            overlay = Image.new("RGB", (width, height), color)
        pil = Image.composite(overlay, pil, alpha)

    return np.asarray(pil, dtype=np.uint8).copy()


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    road_occlusion_probability: float = 0.0,
    road_occlusion_max_patches: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    rotations = random.randrange(4)
    if rotations:
        image, mask = np.rot90(image, rotations), np.rot90(mask, rotations)

    pil = Image.fromarray(np.ascontiguousarray(image))
    if random.random() < 0.60:
        pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.60:
        pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.35:
        pil = ImageEnhance.Color(pil).enhance(random.uniform(0.90, 1.10))
    if random.random() < 0.15:
        pil = pil.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.1)))
    image = np.asarray(pil, dtype=np.uint8).copy()
    if random.random() < 0.15:
        noise = np.random.normal(0.0, random.uniform(2.0, 7.0), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = road_guided_occlusion(
        image,
        mask,
        probability=road_occlusion_probability,
        max_patches=road_occlusion_max_patches,
    )
    return image, np.ascontiguousarray(mask)


def image_to_tensor(image: np.ndarray) -> Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class RoadCropDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        crop_size: int,
        road_crop_probability: float,
        road_crop_min_fraction: float,
        road_crop_tries: int,
        road_occlusion_probability: float,
        road_occlusion_max_patches: int,
    ) -> None:
        self.pairs = list(pairs)
        self.crop_size = int(crop_size)
        self.road_crop_probability = float(road_crop_probability)
        self.road_crop_min_fraction = float(road_crop_min_fraction)
        self.road_crop_tries = int(road_crop_tries)
        self.road_occlusion_probability = float(road_occlusion_probability)
        self.road_occlusion_max_patches = int(road_occlusion_max_patches)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image, mask = read_rgb(image_path), read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch: {image_path} vs {mask_path}")
        image, mask = random_crop_pair(
            image,
            mask,
            self.crop_size,
            self.road_crop_probability,
            self.road_crop_min_fraction,
            self.road_crop_tries,
        )
        image, mask = augment_pair(
            image,
            mask,
            road_occlusion_probability=self.road_occlusion_probability,
            road_occlusion_max_patches=self.road_occlusion_max_patches,
        )
        return image_to_tensor(image), torch.from_numpy(mask).long()


class RoadNativeValidationDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image, mask = read_rgb(image_path), read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch: {image_path} vs {mask_path}")
        return image_to_tensor(image), torch.from_numpy(mask).long(), image_path.stem


def resolve_splits(
    args: argparse.Namespace,
) -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    if args.dataset == "massachusetts":
        train_pairs = pairs_from_list(
            args.train_image_dir, args.train_mask_dir, args.train_list
        )
        listed_test_pairs = pairs_from_list(
            args.train_image_dir, args.train_mask_dir, args.test_list
        )
        train_keys = {sample_key(image) for image, _ in train_pairs}
        listed_test_keys = {sample_key(image) for image, _ in listed_test_pairs}
        overlap = train_keys & listed_test_keys
        if overlap:
            raise RuntimeError(
                f"train.txt and test.txt overlap on {len(overlap)} samples; "
                f"examples={sorted(overlap)[:10]}"
            )
        if args.test_eval_images >= len(listed_test_pairs):
            raise ValueError(
                "test_eval_images must be smaller than the number of test.txt "
                "samples so a non-empty final test split remains"
            )
        val_pairs = listed_test_pairs[: args.test_eval_images]
        test_pairs = listed_test_pairs[args.test_eval_images :]
        rank_zero_print(
            f"TXT split: train={len(train_pairs)}, val={len(val_pairs)}, "
            f"test={len(test_pairs)} | val=first {args.test_eval_images} entries "
            f"of {args.test_list}"
        )
        return train_pairs, val_pairs, test_pairs

    all_pairs = build_pairs(args.train_image_dir, args.train_mask_dir)
    total = len(all_pairs)
    train_count = int(args.deepglobe_train_count)
    val_from_test_count = int(args.deepglobe_val_from_test_count)

    if not 0 < train_count < total:
        raise ValueError(
            f"deepglobe_train_count={train_count} is invalid for {total} labeled pairs"
        )
    test_count = total - train_count
    if not 0 < val_from_test_count <= test_count:
        raise ValueError(
            f"deepglobe_val_from_test_count={val_from_test_count} is invalid for "
            f"a {test_count}-image test split"
        )

    generator = np.random.default_rng(args.split_seed)
    indices = generator.permutation(total)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    train_pairs = [all_pairs[int(i)] for i in train_indices]
    test_pairs = [all_pairs[int(i)] for i in test_indices]
    val_pairs = test_pairs[:val_from_test_count]

    train_keys = {sample_key(image) for image, _ in train_pairs}
    val_keys = {sample_key(image) for image, _ in val_pairs}
    test_keys = {sample_key(image) for image, _ in test_pairs}
    train_test_overlap = train_keys & test_keys
    train_val_overlap = train_keys & val_keys
    val_test_overlap = val_keys & test_keys

    if train_test_overlap:
        raise RuntimeError(
            f"DeepGlobe train/test leakage detected: {len(train_test_overlap)} samples; "
            f"examples={sorted(train_test_overlap)[:10]}"
        )
    if train_val_overlap:
        raise RuntimeError(
            f"DeepGlobe train/val leakage detected: {len(train_val_overlap)} samples; "
            f"examples={sorted(train_val_overlap)[:10]}"
        )
    if len(val_test_overlap) != len(val_pairs):
        raise RuntimeError(
            "DeepGlobe validation must be a subset of the full test split: "
            f"val={len(val_pairs)}, val-test overlap={len(val_test_overlap)}"
        )

    rank_zero_print(
        "DeepGlobe fixed-count split: "
        f"train={len(train_pairs)}, val={len(val_pairs)} (subset of test), "
        f"test={len(test_pairs)} | train-test overlap={len(train_test_overlap)}, "
        f"train-val overlap={len(train_val_overlap)}, "
        f"val-test overlap={len(val_test_overlap)}, seed={args.split_seed}"
    )
    return train_pairs, val_pairs, test_pairs


def make_loaders(
    args: argparse.Namespace,
) -> Tuple[
    DataLoader,
    DataLoader,
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
    Optional[DistributedSampler],
]:
    train_pairs, val_pairs, test_pairs = resolve_splits(args)
    train_dataset = RoadCropDataset(
        train_pairs,
        crop_size=args.crop_size,
        road_crop_probability=args.road_crop_probability,
        road_crop_min_fraction=args.road_crop_min_fraction,
        road_crop_tries=args.road_crop_tries,
        road_occlusion_probability=args.road_occlusion_probability,
        road_occlusion_max_patches=args.road_occlusion_max_patches,
    )
    val_dataset = RoadNativeValidationDataset(val_pairs)
    train_sampler: Optional[DistributedSampler]
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler: Optional[Sampler[int]] = DistributedEvalSampler(
            val_dataset, args.rank, args.world_size
        )
    else:
        train_sampler, val_sampler = None, None
    generator = torch.Generator().manual_seed(args.seed + args.rank)
    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return (
        train_loader,
        val_loader,
        train_pairs,
        val_pairs,
        test_pairs,
        train_sampler,
    )


def compute_road_weight(
    pairs: Sequence[Tuple[Path, Path]], cap: float
) -> Tuple[float, float]:
    positive, total = 0, 0
    report_every = max(1, len(pairs) // 10)
    for index, (_, mask_path) in enumerate(pairs, start=1):
        mask = read_binary_mask(mask_path)
        positive += int(mask.sum())
        total += int(mask.size)
        if is_main_process() and (
            index == 1 or index % report_every == 0 or index == len(pairs)
        ):
            print(
                f"[startup] class-weight scan {index}/{len(pairs)} masks",
                flush=True,
            )
    imbalance = (total - positive) / max(positive, 1)
    return imbalance, min(math.sqrt(imbalance), cap)


def distributed_road_weight(
    pairs: Sequence[Tuple[Path, Path]], cap: float, device: torch.device
) -> Tuple[float, float]:
    values = compute_road_weight(pairs, cap) if is_main_process() else (0.0, 0.0)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed_active():
        dist.broadcast(tensor, src=0)
    return float(tensor[0]), float(tensor[1])


# ---------------------------------------------------------------------------
# Optimizer, progressive unfreezing, EMA, and training
# ---------------------------------------------------------------------------


def _is_broadcast_scale(parameter: torch.nn.Parameter) -> bool:
    """True for per-channel (1, C, 1, 1) scale/gate parameters.

    These behave like a BatchNorm affine weight (an elementwise multiplier),
    not like a dense conv kernel. They should therefore live in the
    ``no_decay`` bucket the same way 1-D biases/norm weights do. Bucketing
    them by ``ndim`` alone (the previous behaviour) puts them in the decay
    group, and weight decay then continuously pulls every learnable fusion
    gate in ``DualResolutionContext`` (``fusion_scale``,
    ``semantic_to_detail_scale_1``, ``detail_to_semantic_scale_1``,
    ``context_scale``) toward zero every step. That is consistent with the
    symptom observed in the Massachusetts run: those gates barely moved
    across 150 epochs and ``detail_to_semantic`` stayed pinned near 0.02.
    """
    return (
        parameter.ndim == 4
        and parameter.shape[0] == 1
        and parameter.shape[2] == 1
        and parameter.shape[3] == 1
    )


def build_optimizer(model: DualBranchRoadNet, args: argparse.Namespace) -> AdamW:
    factors = {
        "head": 1.0,
        "dual_branch": args.dual_branch_lr_factor,
        "layer3": args.backbone_lr_factor,
        "layer2": args.backbone_lr_factor,
        "early_encoder": args.early_encoder_lr_factor,
    }
    groups: List[Dict] = []
    seen: set[int] = set()
    # DEBUG: bucketing summary, printed once at startup so a bad weight-decay
    # assignment (e.g. gate/scale params silently decayed) is visible before
    # a single epoch has run instead of only being inferable from gate drift
    # 100+ epochs later.
    bucket_debug: Dict[str, Dict[str, int]] = {}
    for group_name, parameters in model.optimization_modules().items():
        decay, no_decay = [], []
        scale_like_no_decay = 0
        for parameter in parameters:
            if id(parameter) in seen:
                raise RuntimeError(f"Duplicate optimizer parameter in {group_name}")
            seen.add(id(parameter))
            is_bias_like = parameter.ndim <= 1
            is_scale_like = _is_broadcast_scale(parameter)
            if is_scale_like:
                scale_like_no_decay += 1
            (no_decay if (is_bias_like or is_scale_like) else decay).append(parameter)
        bucket_debug[group_name] = {
            "decay_tensors": len(decay),
            "decay_params": sum(p.numel() for p in decay),
            "no_decay_tensors": len(no_decay),
            "no_decay_params": sum(p.numel() for p in no_decay),
            "scale_like_rescued": scale_like_no_decay,
        }
        for suffix, values, weight_decay in (
            ("decay", decay, args.weight_decay),
            ("no_decay", no_decay, 0.0),
        ):
            if values:
                groups.append(
                    {
                        "params": values,
                        "lr": args.lr * factors[group_name],
                        "weight_decay": weight_decay,
                        "group_name": f"{group_name}/{suffix}",
                    }
                )
    if len(seen) != len(list(model.parameters())):
        raise RuntimeError("Some model parameters were not assigned to the optimizer")

    if is_main_process():
        rank_zero_print("[optimizer] parameter bucketing (decay vs no_decay):")
        for group_name, counts in bucket_debug.items():
            rescued_note = (
                f", scale/gate tensors moved to no_decay={counts['scale_like_rescued']}"
                if counts["scale_like_rescued"]
                else ""
            )
            rank_zero_print(
                f"  {group_name}: decay={counts['decay_tensors']} tensors/"
                f"{counts['decay_params']:,} params | "
                f"no_decay={counts['no_decay_tensors']} tensors/"
                f"{counts['no_decay_params']:,} params"
                f"{rescued_note}"
            )

    optimizer_kwargs = {"betas": (0.9, 0.999)}
    try:
        return AdamW(
            groups,
            fused=bool(torch.cuda.is_available()),
            **optimizer_kwargs,
        )
    except (TypeError, RuntimeError):
        return AdamW(groups, **optimizer_kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    updates_per_epoch: int,
    args: argparse.Namespace,
) -> LambdaLR:
    total = max(1, args.epochs * updates_per_epoch)
    warmup = min(total - 1, max(0, args.warmup_epochs * updates_per_epoch))

    def schedule(update: int) -> float:
        if warmup and update < warmup:
            progress = update / max(1, warmup)
            return args.warmup_start_factor + (
                1.0 - args.warmup_start_factor
            ) * progress
        progress = (update - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return LambdaLR(optimizer, [schedule] * len(optimizer.param_groups))


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.dtype.is_floating_point:
                target.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                target.copy_(value)


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


def phase_for_epoch(epoch: int, args: argparse.Namespace) -> int:
    if not args.progressive_unfreeze:
        return 4
    if epoch < args.unfreeze_dual_branch_epoch:
        return 0
    if epoch < args.unfreeze_layer3_epoch:
        return 1
    if epoch < args.unfreeze_layer2_epoch:
        return 2
    if args.unfreeze_all_epoch < 0 or epoch < args.unfreeze_all_epoch:
        return 3
    return 4


def head_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        if group.get("group_name") == "head/decay":
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def group_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """DEBUG: current LR of the 'decay' sub-group for each optimizer bucket.

    Useful to confirm the per-group LR factors (dual_branch/backbone/etc.)
    and the cosine schedule are actually taking effect as expected, rather
    than inferring it indirectly from the head LR alone.
    """
    lrs: Dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("group_name", "")
        if name.endswith("/decay"):
            lrs[name[: -len("/decay")]] = float(group["lr"])
    return lrs


@torch.no_grad()
def bucket_weight_norms(model: DualBranchRoadNet) -> Dict[str, float]:
    """DEBUG: L2 norm of the trainable weights in each optimizer bucket.

    Tracked every logged epoch so weight-decay shrinkage (or its absence)
    is directly observable instead of only inferable from gate-value drift.
    """
    norms: Dict[str, float] = {}
    for name, parameters in model.optimization_modules().items():
        total = 0.0
        for parameter in parameters:
            total += float(parameter.detach().float().pow(2).sum())
        norms[name] = math.sqrt(total)
    return norms


def bucket_grad_norms(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """DEBUG: L2 grad norm per optimizer bucket, read right after unscale_.

    Gradients are already DDP-averaged and unscaled at this point, so this
    is a cheap way to see whether a given bucket (most importantly
    ``dual_branch``, which holds the bilateral-fusion gates) is receiving a
    meaningfully large gradient at all, independent of whatever the clipped
    global norm ends up being.
    """
    sums: Dict[str, float] = {}
    for group in optimizer.param_groups:
        bucket = group.get("group_name", "unknown").split("/")[0]
        squared = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum())
        sums[bucket] = sums.get(bucket, 0.0) + squared
    return {bucket: math.sqrt(value) for bucket, value in sums.items()}


def train_one_epoch(
    model: nn.Module,
    ema: ModelEMA,
    loader: DataLoader,
    criterion: RoadSegOrientationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    epoch_start = perf_counter()
    model.train()
    base_model = unwrap_model(model)
    base_model.enforce_frozen_norm_eval(args.freeze_encoder_bn)
    if epoch < args.aux_start_epoch:
        aux_scale = 0.0
    elif args.aux_warmup_epochs:
        aux_scale = min(
            1.0,
            (epoch - args.aux_start_epoch + 1) / args.aux_warmup_epochs,
        )
    else:
        aux_scale = 1.0
    criterion.aux_weight = args.aux_weight * aux_scale

    meters = {
        name: RunningAverage()
        for name in (
            "total",
            "main_ce",
            "main_dice",
            "orientation",
            "road_fraction",
            "pred_road_fraction",  # DEBUG: mean predicted road probability
            "grad_norm_total",  # DEBUG: global clipped grad norm
        )
    }
    # DEBUG: per-bucket grad-norm meters, added lazily so this works even if
    # OPTIMIZER_BUCKETS ever drifts from optimization_modules() keys.
    for bucket in OPTIMIZER_BUCKETS:
        meters[f"grad_norm_{bucket}"] = RunningAverage()

    optimizer.zero_grad(set_to_none=True)
    successful_updates, skipped_nonfinite = 0, 0
    progress = tqdm(
        loader,
        desc=f"Train {epoch + 1:03d}",
        leave=False,
        disable=not is_main_process(),
        mininterval=1.0,
    )
    for step, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        group_start = (step // args.accumulation_steps) * args.accumulation_steps
        group_size = min(args.accumulation_steps, len(loader) - group_start)
        do_update = (
            (step + 1) % args.accumulation_steps == 0
            or step + 1 == len(loader)
        )
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and not do_update
            else nullcontext()
        )
        with sync_context:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.use_amp
            ):
                outputs = model(images)
                losses = criterion(outputs, masks)
                scaled_loss = losses["loss_total"] / group_size
            scaler.scale(scaled_loss).backward()

        grad_norm_value = float("nan")
        bucket_norms: Dict[str, float] = {}
        if do_update:
            scaler.unscale_(optimizer)
            total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            grad_norm_value = float(total_norm_tensor)
            if args.debug_grad_norms:
                # Cheap: one Python-side reduction over already-materialized
                # .grad tensors, no extra device synchronization beyond the
                # .item()-style float() casts already required below.
                bucket_norms = bucket_grad_norms(optimizer)
            old_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if float(scaler.get_scale()) >= old_scale:
                scheduler.step()
                ema.update(base_model)
                successful_updates += 1
            else:
                skipped_nonfinite += 1

        batch = int(images.shape[0])
        road_logits = outputs[-1] if isinstance(outputs, tuple) else outputs
        with torch.no_grad():
            pred_road_fraction = (
                road_logits.float().softmax(dim=1)[:, 1].mean()
            )
        metric_values = torch.stack(
            (
                losses["loss_total"].detach(),
                losses["loss_main_ce"],
                losses["loss_main_dice"],
                losses["loss_aux_orientation"],
                (masks > 0).float().mean(),
                pred_road_fraction,
            )
        ).float().cpu().tolist()
        for name, value in zip(
            ("total", "main_ce", "main_dice", "orientation", "road_fraction", "pred_road_fraction"),
            metric_values,
        ):
            meters[name].update(value, batch)
        if do_update and not math.isnan(grad_norm_value):
            meters["grad_norm_total"].update(grad_norm_value, 1)
            for bucket, value in bucket_norms.items():
                if f"grad_norm_{bucket}" in meters:
                    meters[f"grad_norm_{bucket}"].update(value, 1)
        progress.set_postfix(
            loss=f"{meters['total'].mean:.4f}",
            lr=f"{head_lr(optimizer):.2e}",
            aux=f"{criterion.aux_weight:.3f}",
            gnorm=f"{meters['grad_norm_total'].mean:.2f}",
        )

    if distributed_active():
        for meter in meters.values():
            values = torch.tensor(
                [meter.total, meter.count], dtype=torch.float64, device=device
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            meter.total, meter.count = float(values[0]), int(values[1])
        counters = torch.tensor(
            [successful_updates, skipped_nonfinite],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(counters, op=dist.ReduceOp.MAX)
        successful_updates, skipped_nonfinite = map(int, counters.tolist())
    elapsed = max(perf_counter() - epoch_start, 1e-6)
    result = {name: meter.mean for name, meter in meters.items()} | {
        "lr": head_lr(optimizer),
        "group_lrs": group_lrs(optimizer),  # DEBUG
        "aux_weight": criterion.aux_weight,
        "successful_updates": float(successful_updates),
        "skipped_nonfinite": float(skipped_nonfinite),
        "seconds": elapsed,
        "images_per_second": meters["total"].count / elapsed,
    }
    return result


# ---------------------------------------------------------------------------
# Native sliding-window validation and metrics
# ---------------------------------------------------------------------------


def sliding_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def hann_weight(tile_size: int, device: torch.device) -> Tensor:
    axis = torch.hann_window(
        tile_size, periodic=False, dtype=torch.float32, device=device
    ).clamp_min_(0.05)
    return (axis[:, None] * axis[None, :]).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def sliding_window_logits(
    model: nn.Module,
    image: Tensor,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Tensor:
    if image.shape[0] != 1:
        raise ValueError("Native validation requires batch_size=1")
    original_h, original_w = image.shape[-2:]
    pad_h, pad_w = max(0, tile_size - original_h), max(0, tile_size - original_w)
    if pad_h or pad_w:
        mode = "reflect" if min(original_h, original_w) > 1 else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    height, width = image.shape[-2:]
    ys = sliding_positions(height, tile_size, overlap)
    xs = sliding_positions(width, tile_size, overlap)
    coordinates = [(y, x) for y in ys for x in xs]
    accumulator = torch.zeros((1, 2, height, width), device=device)
    normalizer = torch.zeros((1, 1, height, width), device=device)
    weight = hann_weight(tile_size, device)
    for start in range(0, len(coordinates), tile_batch_size):
        batch_coordinates = coordinates[start : start + tile_batch_size]
        tiles = torch.cat(
            [
                image[:, :, y : y + tile_size, x : x + tile_size]
                for y, x in batch_coordinates
            ],
            dim=0,
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(tiles)
        if isinstance(logits, tuple):
            logits = logits[-1]
        logits = logits.float()
        for index, (y, x) in enumerate(batch_coordinates):
            accumulator[:, :, y : y + tile_size, x : x + tile_size] += (
                logits[index : index + 1] * weight
            )
            normalizer[:, :, y : y + tile_size, x : x + tile_size] += weight
    return (accumulator / normalizer.clamp_min_(1e-6))[
        :, :, :original_h, :original_w
    ]


def confusion_counts(prediction: Tensor, target: Tensor) -> Tuple[int, int, int, int]:
    prediction, target = prediction.bool(), target.bool()
    return (
        int((prediction & target).sum()),
        int((prediction & ~target).sum()),
        int((~prediction & target).sum()),
        int((~prediction & ~target).sum()),
    )


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    road_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "road_iou": road_iou,
        "background_iou": background_iou,
        "miou": 0.5 * (road_iou + background_iou),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def histogram_counts(
    probability: Tensor, target: Tensor, bins: int
) -> Tuple[Tensor, Tensor]:
    indices = (probability.clamp(0, 1) * (bins - 1)).long().flatten()
    labels = target.bool().flatten()
    positive = torch.bincount(indices[labels], minlength=bins)
    negative = torch.bincount(indices[~labels], minlength=bins)
    return positive, negative


def counts_at_threshold(
    positive: Tensor, negative: Tensor, threshold: float
) -> Tuple[int, int, int, int]:
    boundary = int(math.ceil(threshold * (len(positive) - 1)))
    tp = int(positive[boundary:].sum())
    fp = int(negative[boundary:].sum())
    fn = int(positive[:boundary].sum())
    tn = int(negative[:boundary].sum())
    return tp, fp, fn, tn


def relaxed_components(
    prediction: Tensor, target: Tensor, buffer_px: int
) -> Tuple[float, float, float, float]:
    pred = prediction.float().unsqueeze(0).unsqueeze(0)
    truth = target.float().unsqueeze(0).unsqueeze(0)
    kernel = 2 * buffer_px + 1
    pred_dilated = F.max_pool2d(pred, kernel, stride=1, padding=buffer_px) > 0
    truth_dilated = F.max_pool2d(truth, kernel, stride=1, padding=buffer_px) > 0
    return (
        float((prediction & truth_dilated[0, 0]).sum()),
        float((target & pred_dilated[0, 0]).sum()),
        float(prediction.sum()),
        float(target.sum()),
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    road_class_weight: float,
    main_dice_weight: float,
) -> Dict[str, float]:
    model.eval()
    bins = args.threshold_bins
    positive_hist = torch.zeros(bins, dtype=torch.int64, device=device)
    negative_hist = torch.zeros(bins, dtype=torch.int64, device=device)
    # Slots 0-9 are the pre-existing confusion/relaxed/count accumulators;
    # 10-11 are the running main-loss sums added for train/val overfitting
    # comparison (see val_loss_main below). Appending at the end keeps every
    # existing totals[...] index below unchanged.
    totals = torch.zeros(12, dtype=torch.float64, device=device)
    # DEBUG: probability-mass diagnostics to separate "under-confident" from
    # "wrong" predictions -- summed then normalized after the loop.
    probability_sum_on_road = torch.zeros((), dtype=torch.float64, device=device)
    probability_sum_on_background = torch.zeros((), dtype=torch.float64, device=device)
    road_pixel_count = torch.zeros((), dtype=torch.float64, device=device)
    background_pixel_count = torch.zeros((), dtype=torch.float64, device=device)
    progress = tqdm(
        loader, desc="Native validation", leave=False, disable=not is_main_process()
    )
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        logits = sliding_window_logits(
            model,
            images,
            args.val_tile_size,
            args.val_overlap,
            args.val_tile_batch_size,
            device,
            args.use_amp,
        )
        probability = logits.softmax(dim=1)[:, 1]
        target = masks > 0
        prediction = probability >= 0.5
        tp, fp, fn, tn = confusion_counts(prediction, target)
        per_image_iou = tp / max(tp + fp + fn, 1)
        relaxed = relaxed_components(
            prediction[0], target[0], args.relaxed_buffer_px
        )
        # DEBUG: main-loss-only (CE + Dice) validation loss, for comparing
        # against train_metrics["total"] to spot overfitting. The orientation
        # auxiliary term is excluded because RoadReconstructionDecoder only
        # returns OrientedSkipAggregation's prediction in training mode
        # (self.training), and running the model in train() mode here would
        # corrupt BatchNorm running stats / enable dropout -- not an
        # acceptable trade for one extra loss term.
        labels = target.long()
        class_weights = logits.new_tensor([1.0, road_class_weight])
        loss_main_ce = F.cross_entropy(
            logits.float(), labels, weight=class_weights
        )
        loss_main_dice = binary_dice_loss(
            probability.unsqueeze(1), labels.unsqueeze(1).float()
        )
        totals += totals.new_tensor(
            [
                tp, fp, fn, tn, per_image_iou, 1.0, *relaxed,
                float(loss_main_ce.detach()), float(loss_main_dice.detach()),
            ]
        )
        positive, negative = histogram_counts(probability, target, bins)
        positive_hist += positive
        negative_hist += negative

        probability_sum_on_road += probability[target].double().sum()
        probability_sum_on_background += probability[~target].double().sum()
        road_pixel_count += target.double().sum()
        background_pixel_count += (~target).double().sum()

    if distributed_active():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(positive_hist, op=dist.ReduceOp.SUM)
        dist.all_reduce(negative_hist, op=dist.ReduceOp.SUM)
        for tensor in (
            probability_sum_on_road,
            probability_sum_on_background,
            road_pixel_count,
            background_pixel_count,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tp, fp, fn, tn = (int(value) for value in totals[:4].tolist())
    metrics = {f"fixed_{key}": value for key, value in metrics_from_counts(tp, fp, fn, tn).items()}
    metrics["fixed_road_iou_macro"] = float(totals[4] / max(float(totals[5]), 1.0))
    relaxed_precision = float(totals[6] / max(float(totals[8]), 1.0))
    relaxed_recall = float(totals[7] / max(float(totals[9]), 1.0))
    metrics["fixed_relaxed_f1"] = 2.0 * relaxed_precision * relaxed_recall / max(
        relaxed_precision + relaxed_recall, 1e-12
    )
    # DEBUG: mean predicted probability, split by true class. A model that is
    # merely under-confident (mean prob on road pixels well below 0.5, but
    # separated from background) benefits from threshold calibration; a
    # model that is genuinely confused has both means close together.
    metrics["mean_probability_on_road"] = float(
        probability_sum_on_road / road_pixel_count.clamp_min(1.0)
    )
    metrics["mean_probability_on_background"] = float(
        probability_sum_on_background / background_pixel_count.clamp_min(1.0)
    )
    metrics["predicted_road_fraction"] = float(
        (tp + fp) / max(tp + fp + fn + tn, 1)
    )
    metrics["true_road_fraction"] = float(
        (tp + fn) / max(tp + fp + fn + tn, 1)
    )

    best_threshold, best_counts, best_iou = 0.5, (tp, fp, fn, tn), -1.0
    threshold = args.threshold_min
    while threshold <= args.threshold_max + 1e-9:
        counts = counts_at_threshold(positive_hist, negative_hist, threshold)
        candidate = metrics_from_counts(*counts)["road_iou"]
        if candidate > best_iou:
            best_threshold, best_counts, best_iou = threshold, counts, candidate
        threshold += args.threshold_step
    metrics.update(
        {
            f"calibrated_{key}": value
            for key, value in metrics_from_counts(*best_counts).items()
        }
    )
    metrics["calibrated_threshold"] = float(best_threshold)

    image_count = max(float(totals[5]), 1.0)
    metrics["val_loss_main_ce"] = float(totals[10] / image_count)
    metrics["val_loss_main_dice"] = float(totals[11] / image_count)
    metrics["val_loss_main"] = (
        metrics["val_loss_main_ce"]
        + float(main_dice_weight) * metrics["val_loss_main_dice"]
    )
    return metrics


# ---------------------------------------------------------------------------
# Checkpoints and argument handling
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        preferred = (
            "best_fixed_road_iou.pt",
            "best_calibrated_road_iou.pt",
            "last.pt",
        )
        for name in preferred:
            candidate = path / name
            if candidate.is_file():
                return candidate
        files = sorted(path.rglob("*.pt")) + sorted(path.rglob("*.pth"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(f"Checkpoint not found or ambiguous: {path}")


def safe_torch_load(path: str | Path, device: torch.device):
    path = resolve_checkpoint_path(path)
    try:
        return path, torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return path, torch.load(path, map_location=device)


def clean_state_dict(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def transfer_weights(
    model: nn.Module,
    path: str | Path,
    weights: str,
    device: torch.device,
) -> Path:
    checkpoint_path, checkpoint = safe_torch_load(path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Transfer checkpoint must be a dictionary")
    state = checkpoint.get(weights)
    if not isinstance(state, dict):
        fallback = "model" if weights == "ema" else "ema"
        state = checkpoint.get(fallback, checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise KeyError(f"No '{weights}', model, ema, or state_dict weights found")
    cleaned = clean_state_dict(state)
    result = model.load_state_dict(cleaned, strict=False)
    # NEW_MODULE_NAME_FRAGMENTS: substrings of parameter names that are
    # allowed to be absent from an older checkpoint because they belong to a
    # module introduced after that checkpoint was trained. Extend this tuple
    # (never remove from it) whenever a new purely-additive, zero/near-zero
    # initialized submodule is added, so older checkpoints keep transferring.
    new_module_fragments = ("spatial_gate", "skip_gate", "full_extra_refine")
    allowed_missing = all(
        any(fragment in key for fragment in new_module_fragments)
        for key in result.missing_keys
    )
    if result.unexpected_keys or not allowed_missing:
        raise RuntimeError(
            "Transfer checkpoint architecture does not match DualBranchRoadNet. "
            f"Missing={result.missing_keys}, unexpected={result.unexpected_keys}. "
            f"Only tensors matching {new_module_fragments} may be absent."
        )
    return checkpoint_path


def checkpoint_state(
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    epoch: int,
    best_fixed: float,
    best_calibrated: float,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Dict:
    return {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema.module.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_fixed_road_iou": best_fixed,
        "best_calibrated_road_iou": best_calibrated,
        "validation": metrics,
        "args": vars(args),
    }


def atomic_torch_save(state: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume_training(
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    path: str | Path,
    device: torch.device,
) -> Tuple[int, float, float, Path]:
    checkpoint_path, checkpoint = safe_torch_load(path, device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    ema.module.load_state_dict(clean_state_dict(checkpoint["ema"]), strict=True)
    ema.updates = int(checkpoint.get("ema_updates", 0))
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_fixed_road_iou", -1.0)),
        float(checkpoint.get("best_calibrated_road_iou", -1.0)),
        checkpoint_path,
    )


def append_jsonl(path: str | Path, record: Dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=("deepglobe", "massachusetts"), required=True
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--train_image_dir", default=None)
    parser.add_argument("--train_mask_dir", default=None)
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--test_list", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_mask_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="Held out and never evaluated during training when no labeled val exists",
    )
    parser.add_argument("--split_seed", type=int, default=3407)
    parser.add_argument(
        "--deepglobe_train_count",
        type=int,
        default=5000,
        help=(
            "DeepGlobe fixed training count. With the 6226 labeled pairs, "
            "the remaining 1226 images form the full test split."
        ),
    )
    parser.add_argument(
        "--deepglobe_val_from_test_count",
        type=int,
        default=300,
        help=(
            "Number of validation images taken deterministically from the start "
            "of the full DeepGlobe test1226 split."
        ),
    )

    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--road_crop_probability", type=float, default=0.60)
    parser.add_argument("--road_crop_min_fraction", type=float, default=0.002)
    parser.add_argument("--road_crop_tries", type=int, default=8)
    parser.add_argument(
        "--road_occlusion_probability",
        type=float,
        default=0.0,
        help="Probability of placing short synthetic occluders on road pixels",
    )
    parser.add_argument(
        "--road_occlusion_max_patches",
        type=int,
        default=2,
    )

    parser.add_argument("--detail_channels", type=int, default=96)
    parser.add_argument("--semantic_channels", type=int, default=192)
    parser.add_argument("--dappm_channels", type=int, default=32)
    parser.add_argument(
        "--dappm_pool_sizes", nargs="+", type=int, default=(1, 2, 4, 8)
    )
    parser.add_argument(
        "--detail_blocks", nargs=2, type=int, default=(2, 2)
    )
    parser.add_argument("--semantic_blocks", type=int, default=2)
    parser.add_argument("--fusion_blocks", type=int, default=1)
    parser.add_argument(
        "--bilateral_fusion",
        choices=("static", "spatial"),
        default="spatial",
        help=(
            "static reproduces the original channel-scaled residual exchange; "
            "spatial adds lightweight location-adaptive gates in both directions"
        ),
    )
    parser.add_argument("--decoder_s4_channels", type=int, default=64)
    parser.add_argument("--decoder_s2_channels", type=int, default=32)
    parser.add_argument("--full_channels", type=int, default=24)
    parser.add_argument(
        "--full_refine_blocks",
        type=int,
        default=0,
        help=(
            "Extra RepDepthwiseBlocks after full_refine, at full crop "
            "resolution (the most expensive spatial size in the decoder -- "
            "each block here costs far more compute/VRAM than the same "
            "block at S4/S2, despite full_channels usually being the "
            "smallest width). Default 0 keeps the original single-block "
            "full_refine so a plain run fits the same batch size as before; "
            "opt in only once there is VRAM budget to test it."
        ),
    )
    parser.add_argument(
        "--oriented_skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use OrientedSkipAggregation (road-direction-steered context "
            "aggregation, no auxiliary loss) for the S4/S2 skip connections "
            "instead of SkipFeatureGate (channel/spatial SE-style gate). "
            "--no-oriented_skip restores SkipFeatureGate for comparison."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--imagenet_pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--encoder_weights_path",
        default=None,
        help="Local torchvision-format ResNet-34 weights; avoids downloading",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2, help="Per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4, help="Decoder/head LR")
    parser.add_argument(
        "--dual_branch_lr_factor",
        "--bottleneck_lr_factor",
        dest="dual_branch_lr_factor",
        type=float,
        default=0.75,
        help="LR factor for detail/semantic branches; old flag kept as alias",
    )
    parser.add_argument("--backbone_lr_factor", type=float, default=0.20)
    parser.add_argument("--early_encoder_lr_factor", type=float, default=0.10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--road_weight_cap", type=float, default=2.0)
    parser.add_argument(
        "--fixed_road_weight",
        type=float,
        default=None,
        help="Positive value skips the startup mask scan and uses this CE weight",
    )
    parser.add_argument("--main_dice_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.15)
    parser.add_argument(
        "--aux_start_epoch",
        type=int,
        default=5,
        help="Keep orientation supervision off before this zero-based epoch",
    )
    parser.add_argument("--aux_warmup_epochs", type=int, default=5)

    parser.add_argument(
        "--progressive_unfreeze",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled for transfer runs, disabled otherwise",
    )
    parser.add_argument(
        "--unfreeze_dual_branch_epoch",
        "--unfreeze_bottleneck_epoch",
        dest="unfreeze_dual_branch_epoch",
        type=int,
        default=3,
        help="Epoch index opening both detail and semantic branches",
    )
    parser.add_argument("--unfreeze_layer3_epoch", type=int, default=6)
    parser.add_argument("--unfreeze_layer2_epoch", type=int, default=12)
    parser.add_argument(
        "--unfreeze_all_epoch",
        type=int,
        default=-1,
        help="Negative keeps ResNet stem/layer1 frozen for the whole transfer run",
    )
    parser.add_argument(
        "--freeze_encoder_bn", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--val_tile_size", type=int, default=1024)
    parser.add_argument("--val_overlap", type=int, default=256)
    parser.add_argument("--val_tile_batch_size", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--test_eval_images",
        type=int,
        default=61,
        help=(
            "For the Massachusetts txt protocol, reserve the first N test.txt "
            "entries for validation and keep the remainder for final testing"
        ),
    )
    parser.add_argument("--threshold_min", type=float, default=0.20)
    parser.add_argument("--threshold_max", type=float, default=0.80)
    parser.add_argument("--threshold_step", type=float, default=0.02)
    parser.add_argument("--threshold_bins", type=int, default=1001)
    parser.add_argument("--relaxed_buffer_px", type=int, default=3)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--channels_last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--transfer_weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_dir", default="./checkpoints/dual_branch_roadnet")

    # ---- DEBUG / diagnostics flags -----------------------------------
    parser.add_argument(
        "--debug_grad_norms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-bucket gradient L2 norms every epoch (head/dual_branch/"
        "layer3/layer2/early_encoder). Cheap (no extra device sync).",
    )
    parser.add_argument(
        "--debug_gate_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print DualResolutionContext gate statistics every epoch "
        "instead of only on validation epochs.",
    )
    parser.add_argument(
        "--debug_weight_norms_every",
        type=int,
        default=10,
        help="Print per-bucket trainable weight L2 norm every N epochs "
        "(0 disables). Useful to directly see weight-decay shrinkage.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.crop_size < 64 or args.crop_size % 16:
        raise ValueError("crop_size must be >=64 and divisible by 16")
    if args.val_overlap < 0 or args.val_overlap >= args.val_tile_size:
        raise ValueError("val_overlap must satisfy 0 <= overlap < tile size")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("batch_size and accumulation_steps must be positive")
    if args.dataset == "massachusetts":
        if not args.train_list or not args.test_list:
            raise ValueError("Massachusetts requires --train_list and --test_list")
        if args.test_eval_images < 1:
            raise ValueError("test_eval_images must be positive")
    elif args.dataset == "deepglobe":
        if args.deepglobe_train_count < 1:
            raise ValueError("deepglobe_train_count must be positive")
        if args.deepglobe_val_from_test_count < 1:
            raise ValueError("deepglobe_val_from_test_count must be positive")
    if args.fixed_road_weight is not None and args.fixed_road_weight <= 0.0:
        raise ValueError("fixed_road_weight must be positive")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    if not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be smaller than 1")
    if args.resume and args.pretrained_checkpoint:
        raise ValueError("Use either --resume or --pretrained_checkpoint, not both")
    if not args.dappm_pool_sizes or min(args.dappm_pool_sizes) < 1:
        raise ValueError("dappm_pool_sizes must be positive")
    if min(args.detail_blocks) < 1:
        raise ValueError("detail_blocks must be positive")
    channel_values = (
        args.detail_channels,
        args.semantic_channels,
        args.dappm_channels,
        args.decoder_s4_channels,
        args.decoder_s2_channels,
        args.full_channels,
    )
    if min(channel_values) < 1:
        raise ValueError("All architecture channel counts must be positive")
    if not 0.0 <= args.road_occlusion_probability <= 1.0:
        raise ValueError("road_occlusion_probability must be in [0, 1]")
    if args.road_occlusion_max_patches < 1:
        raise ValueError("road_occlusion_max_patches must be positive")
    for name in ("aux_weight",):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} cannot be negative")
    if args.aux_start_epoch < 0 or args.aux_warmup_epochs < 0:
        raise ValueError("Centerline start/warmup epochs cannot be negative")
    if args.progressive_unfreeze:
        epochs = (
            args.unfreeze_dual_branch_epoch,
            args.unfreeze_layer3_epoch,
            args.unfreeze_layer2_epoch,
        )
        if min(epochs) < 0 or tuple(sorted(epochs)) != epochs:
            raise ValueError("Unfreeze epochs must be non-negative and ordered")
        if 0 <= args.unfreeze_all_epoch < args.unfreeze_layer2_epoch:
            raise ValueError("unfreeze_all_epoch must follow unfreeze_layer2_epoch")


def save_split_manifest(
    save_dir: Path,
    train_pairs: Sequence[Tuple[Path, Path]],
    val_pairs: Sequence[Tuple[Path, Path]],
    test_pairs: Sequence[Tuple[Path, Path]],
    split_seed: int,
) -> None:
    train_keys = {sample_key(image) for image, _ in train_pairs}
    val_keys = {sample_key(image) for image, _ in val_pairs}
    test_keys = {sample_key(image) for image, _ in test_pairs}
    overlap_counts = {
        "train_val": len(train_keys & val_keys),
        "train_test": len(train_keys & test_keys),
        "val_test": len(val_keys & test_keys),
    }
    manifest = {
        "split_seed": int(split_seed),
        "counts": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "overlap_counts": overlap_counts,
        "train": [[str(image), str(mask)] for image, mask in train_pairs],
        "val": [[str(image), str(mask)] for image, mask in val_pairs],
        "test": [[str(image), str(mask)] for image, mask in test_pairs],
    }
    with (save_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed()
    args.distributed, args.rank = distributed, rank
    args.local_rank, args.world_size = local_rank, world_size
    if args.progressive_unfreeze is None:
        if args.resume:
            _, resume_metadata = safe_torch_load(
                args.resume, torch.device("cpu")
            )
            saved_args = (
                resume_metadata.get("args", {})
                if isinstance(resume_metadata, dict)
                else {}
            )
            args.progressive_unfreeze = bool(
                saved_args.get("progressive_unfreeze", False)
            )
        else:
            args.progressive_unfreeze = args.pretrained_checkpoint is not None
    configure_dataset_paths(args)
    validate_args(args)
    rank_zero_print(
        f"[startup 1/5] DDP initialized: world_size={world_size}, device={device}"
    )
    seed_everything(args.seed + rank)
    args.use_amp = bool(args.use_amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    save_dir = Path(args.save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
    rank_zero_print("[startup 2/5] Resolving image/mask pairs and DataLoaders...")
    (
        train_loader,
        val_loader,
        train_pairs,
        val_pairs,
        test_pairs,
        train_sampler,
    ) = make_loaders(args)
    if is_main_process():
        save_split_manifest(
            save_dir,
            train_pairs,
            val_pairs,
            test_pairs,
            args.split_seed,
        )
        if args.dataset == "deepglobe":
            rank_zero_print(
                f"[startup] split_manifest.json saved: train={len(train_pairs)}, "
                f"val={len(val_pairs)}, test={len(test_pairs)}; "
                "expected overlaps train-test=0, train-val=0, val-test=300"
            )
    if args.fixed_road_weight is None:
        rank_zero_print(
            f"[startup 3/5] Scanning {len(train_pairs)} training masks "
            "to estimate the road class weight..."
        )
        imbalance, road_weight = distributed_road_weight(
            train_pairs, args.road_weight_cap, device
        )
    else:
        imbalance = float("nan")
        road_weight = float(args.fixed_road_weight)
        rank_zero_print(
            f"[startup 3/5] Skipping mask scan; fixed road CE weight="
            f"{road_weight:.3f}"
        )

    build_args = copy.copy(args)
    if args.pretrained_checkpoint or args.resume:
        build_args.imagenet_pretrained = False

    needs_imagenet_cache = bool(
        build_args.imagenet_pretrained and not build_args.encoder_weights_path
    )
    rank_zero_print(
        "[startup 4/5] Building DualBranchRoadNet"
        + (
            " and loading/downloading ImageNet ResNet-34 weights..."
            if needs_imagenet_cache
            else "..."
        )
    )
    if distributed and rank != 0 and needs_imagenet_cache:
        dist.barrier(device_ids=[local_rank])
    model = build_model(build_args).to(device)
    if distributed and rank == 0 and needs_imagenet_cache:
        dist.barrier(device_ids=[local_rank])
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.pretrained_checkpoint:
        loaded = transfer_weights(
            model, args.pretrained_checkpoint, args.transfer_weights, device
        )
        rank_zero_print(f"Transferred {args.transfer_weights} weights from {loaded}")
    rank_zero_print("[startup 5/5] Building optimizer, EMA, loss, and DDP reducer...")

    model.set_trainable_phase(4)
    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = build_scheduler(optimizer, updates_per_epoch, args)
    scaler = make_grad_scaler(args.use_amp)
    ema = ModelEMA(model, args.ema_decay)
    criterion = RoadSegOrientationLoss(
        road_class_weight=road_weight,
        main_dice_weight=args.main_dice_weight,
        aux_weight=args.aux_weight,
    ).to(device)

    start_epoch, best_fixed, best_calibrated = 0, -1.0, -1.0
    if args.resume:
        start_epoch, best_fixed, best_calibrated, loaded = resume_training(
            model, ema, optimizer, scheduler, scaler, args.resume, device
        )
        rank_zero_print(f"Resumed exact training state from {loaded}")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.progressive_unfreeze),
            gradient_as_bucket_view=True,
            broadcast_buffers=True,
        )

    effective_batch = args.batch_size * world_size * args.accumulation_steps
    rank_zero_print("=" * 78)
    rank_zero_print("DualBranchRoadNet road training")
    rank_zero_print(
        f"dataset={args.dataset} | DDP={distributed} | GPUs={world_size} | "
        f"AMP={args.use_amp} | channels_last={args.channels_last}"
    )
    rank_zero_print(
        f"crop={args.crop_size} | batch/GPU={args.batch_size} | "
        f"accumulation={args.accumulation_steps} | effective batch={effective_batch}"
    )
    rank_zero_print(
        f"shared ResNet34 to S8 | detail={args.detail_channels}ch S8 | "
        f"semantic anchor=256ch S16 | context={args.semantic_channels}ch S32 | DAPPM="
        f"{args.dappm_channels}ch grids={tuple(args.dappm_pool_sizes)}"
    )
    rank_zero_print(
        f"decoder S4/S2/S1={args.decoder_s4_channels}/"
        f"{args.decoder_s2_channels}/{args.full_channels}ch | "
        f"detail blocks={tuple(args.detail_blocks)} | "
        f"full_refine_blocks={args.full_refine_blocks} | "
        f"skip={'oriented' if args.oriented_skip else 'gate'}"
    )
    rank_zero_print(
        f"bilateral fusion={args.bilateral_fusion} | "
        "loss and directional decoder unchanged"
    )
    rank_zero_print(
        f"parameters={total_parameters:,} | imbalance={imbalance:.3f} | "
        f"road CE weight={road_weight:.3f}"
    )
    rank_zero_print(
        "loss=weighted CE + Dice"
        + (
            " + road-orientation supervision (S2 OrientedSkipAggregation)"
            if args.oriented_skip
            else ""
        )
        + f" (aux max={args.aux_weight:.2f}, starts epoch "
        f"{args.aux_start_epoch + 1})"
    )
    rank_zero_print(
        f"progressive_unfreeze={args.progressive_unfreeze} | "
        f"epochs={args.epochs} | head LR={args.lr:.2e}"
    )
    rank_zero_print("=" * 78)

    log_path = save_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        phase = phase_for_epoch(epoch, args)
        phase_name = unwrap_model(model).set_trainable_phase(phase)
        trainable, total = unwrap_model(model).trainable_parameter_counts()
        rank_zero_print(
            f"\nEpoch {epoch + 1}/{args.epochs} | phase={phase_name} | "
            f"trainable={trainable:,}/{total:,}"
        )
        train_metrics = train_one_epoch(
            model,
            ema,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            args,
        )
        gate_metrics = unwrap_model(model).dual_branch.gate_statistics()

        # DEBUG: print a compact per-epoch line every epoch, not only on
        # validation epochs. This is what lets you see gate/grad behaviour
        # at 1-epoch resolution instead of only every val_interval epochs.
        if is_main_process() and args.debug_gate_every_epoch:
            group_lr_str = " ".join(
                f"{name}={lr:.2e}" for name, lr in train_metrics["group_lrs"].items()
            )
            grad_norm_str = " ".join(
                f"{bucket}={train_metrics.get(f'grad_norm_{bucket}', float('nan')):.3f}"
                for bucket in OPTIMIZER_BUCKETS
            )
            rank_zero_print(
                f"  [debug] train_loss={train_metrics['total']:.5f} "
                f"(ce={train_metrics['main_ce']:.4f} "
                f"dice={train_metrics['main_dice']:.4f} "
                f"orientation={train_metrics['orientation']:.4f}) | "
                f"pred_road_frac={train_metrics['pred_road_fraction']:.4f} "
                f"true_road_frac={train_metrics['road_fraction']:.4f} | "
                f"skipped_nonfinite={int(train_metrics['skipped_nonfinite'])}"
            )
            rank_zero_print(f"  [debug] group LRs: {group_lr_str}")
            if args.debug_grad_norms:
                rank_zero_print(
                    f"  [debug] grad norms (total={train_metrics['grad_norm_total']:.3f}): "
                    f"{grad_norm_str}"
                )
            rank_zero_print(
                "  [debug] fusion gates s2d/d2s/ctx/final="
                f"{gate_metrics['semantic_to_detail_abs_mean']:.4f}/"
                f"{gate_metrics['detail_to_semantic_abs_mean']:.4f}/"
                f"{gate_metrics['s32_context_to_s16_abs_mean']:.4f}/"
                f"{gate_metrics['semantic_to_final_abs_mean']:.4f}"
                + (
                    " | spatial mean(std) s2d/d2s="
                    f"{gate_metrics['semantic_to_detail_spatial_mean']:.4f}"
                    f"({gate_metrics['semantic_to_detail_spatial_std']:.4f})/"
                    f"{gate_metrics['detail_to_semantic_spatial_mean']:.4f}"
                    f"({gate_metrics['detail_to_semantic_spatial_std']:.4f})"
                    if args.bilateral_fusion == "spatial"
                    else ""
                )
            )
            if (
                args.debug_weight_norms_every
                and (epoch + 1) % args.debug_weight_norms_every == 0
            ):
                weight_norms = bucket_weight_norms(unwrap_model(model))
                weight_norm_str = " ".join(
                    f"{name}={norm:.2f}" for name, norm in weight_norms.items()
                )
                rank_zero_print(f"  [debug] bucket weight L2 norms: {weight_norm_str}")

        validation_metrics: Dict[str, float] = {}
        should_validate = (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs
        if should_validate:
            if distributed_active():
                for tensor in ema.module.state_dict().values():
                    dist.broadcast(tensor, src=0)
            validation_metrics = validate(
                ema.module,
                val_loader,
                device,
                args,
                criterion.road_class_weight,
                criterion.main_dice_weight,
            )
            fixed = validation_metrics["fixed_road_iou"]
            calibrated = validation_metrics["calibrated_road_iou"]
            fixed_improved, calibrated_improved = fixed > best_fixed, calibrated > best_calibrated
            best_fixed, best_calibrated = max(best_fixed, fixed), max(best_calibrated, calibrated)
            # DEBUG: train vs. val main loss (CE+Dice; train loss below also
            # includes the orientation term, val_loss_main does not -- see
            # its computation site in validate() for why it can't). A train
            # loss that keeps falling while val_loss_main flattens or rises
            # is the overfitting signal to watch for.
            rank_zero_print(
                f"train loss={train_metrics['total']:.5f} | "
                f"val loss={validation_metrics['val_loss_main']:.5f} "
                f"(ce={validation_metrics['val_loss_main_ce']:.4f} "
                f"dice={validation_metrics['val_loss_main_dice']:.4f}) | "
                f"throughput={train_metrics['images_per_second']:.1f} img/s | "
                f"fixed@.50 road IoU={fixed:.5f} | "
                f"calibrated road IoU={calibrated:.5f} "
                f"@{validation_metrics['calibrated_threshold']:.2f} | "
                f"F1={validation_metrics['fixed_f1']:.5f} | "
                f"gates s2d/d2s/ctx/final="
                f"{gate_metrics['semantic_to_detail_abs_mean']:.3f}/"
                f"{gate_metrics['detail_to_semantic_abs_mean']:.3f}/"
                f"{gate_metrics['s32_context_to_s16_abs_mean']:.3f}/"
                f"{gate_metrics['semantic_to_final_abs_mean']:.3f}"
                + (
                    " | spatial mean s2d/d2s="
                    f"{gate_metrics['semantic_to_detail_spatial_mean']:.3f}/"
                    f"{gate_metrics['detail_to_semantic_spatial_mean']:.3f}"
                    " std="
                    f"{gate_metrics['semantic_to_detail_spatial_std']:.3f}/"
                    f"{gate_metrics['detail_to_semantic_spatial_std']:.3f}"
                    if args.bilateral_fusion == "spatial"
                    else ""
                )
            )
            # DEBUG: calibration / confusion diagnostics, printed every
            # validation epoch so under-confidence vs. genuine confusion is
            # visible without post-hoc log parsing.
            rank_zero_print(
                "  [debug] precision/recall(fixed)="
                f"{validation_metrics['fixed_precision']:.4f}/"
                f"{validation_metrics['fixed_recall']:.4f} | "
                f"relaxed_f1={validation_metrics['fixed_relaxed_f1']:.4f} | "
                f"mean_prob(road/bg)="
                f"{validation_metrics['mean_probability_on_road']:.4f}/"
                f"{validation_metrics['mean_probability_on_background']:.4f} | "
                f"pred_road_frac/true_road_frac="
                f"{validation_metrics['predicted_road_fraction']:.4f}/"
                f"{validation_metrics['true_road_fraction']:.4f}"
            )
            if is_main_process():
                state = checkpoint_state(
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_fixed,
                    best_calibrated,
                    validation_metrics,
                    args,
                )
                if fixed_improved:
                    atomic_torch_save(state, save_dir / "best_fixed_road_iou.pt")
                if calibrated_improved:
                    atomic_torch_save(
                        state, save_dir / "best_calibrated_road_iou.pt"
                    )
                atomic_torch_save(state, save_dir / "last.pt")
        elif is_main_process():
            state = checkpoint_state(
                model,
                ema,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_fixed,
                best_calibrated,
                validation_metrics,
                args,
            )
            atomic_torch_save(state, save_dir / "last.pt")

        if is_main_process():
            append_jsonl(
                log_path,
                {
                    "epoch": epoch + 1,
                    "phase": phase_name,
                    "trainable_parameters": trainable,
                    "train": train_metrics,
                    "fusion_gates": gate_metrics,
                    "validation": validation_metrics,
                },
            )
    rank_zero_print(f"Finished. Checkpoints: {save_dir.resolve()}")
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
