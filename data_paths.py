"""Read the Massachusetts / DeepGlobe road datasets from the data/ folder.

Users only drop the downloaded files into four folders:

  data/massachusetts/train/    data/massachusetts/test/
  data/deepglobe/train/        data/deepglobe/test/

Inside a train/ or test/ folder the layout does not matter (flat, or split into
images/ and labels/ sub-folders). Files are classified as image or label and
then paired by file name without extension; files with no partner are ignored.

A file is a label when its name ends with _mask, _masks, _gt, _label or
_labels, or when a folder between train/ (or test/) and the file has one of
label(s), mask(s), gt, groundtruth, annotation(s) in its name.

train.py and test_native.py both read data through this module. ``--data_root``
replaces data/<dataset>/ with another folder that has the same train/ and
test/ sub-folders.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
MASK_DIR_WORDS = {
    "label", "labels", "mask", "masks", "gt", "groundtruth",
    "annotation", "annotations",
}
SPLITS = ("train", "test")

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
            "images and their labels inside (see data/README.md)."
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
            "folder (see data/README.md)."
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
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = Path(data_root).expanduser() if data_root else DATA_DIR / dataset
    return collect_pairs(root / split)


def load_pairs(
    dataset: str, data_root: Optional[str | Path] = None
) -> Tuple[List[Pair], List[Pair]]:
    """(train_pairs, test_pairs) for ``dataset``, checked to be disjoint."""
    train = load_split(dataset, "train", data_root)
    test = load_split(dataset, "test", data_root)
    overlap = {sample_key(i) for i, _ in train} & {sample_key(i) for i, _ in test}
    if overlap:
        raise RuntimeError(
            f"train/ and test/ share {len(overlap)} samples; "
            f"examples={sorted(overlap)[:10]}"
        )
    return train, test
