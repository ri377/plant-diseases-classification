"""Create a small balanced subset of the New Plant Diseases dataset.

Use this once to produce a ~50 MB sample that you can DVC-track for fast local
development. Defaults pick 10 random classes and copy 30 train + 8 valid images each
(so the model still trains end-to-end, just badly).

Usage:
    python scripts/make_subset.py \
        --source data/new-plant-diseases-dataset \
        --output data/new-plant-diseases-dataset-subset
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import fire

INNER_NEST = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"


def make_subset(
    source: str = "data/new-plant-diseases-dataset",
    output: str = "data/new-plant-diseases-dataset-subset",
    classes_per_subset: int = 10,
    train_per_class: int = 30,
    val_per_class: int = 8,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    source = Path(source)
    output = Path(output)

    train_src = source / INNER_NEST / "train"
    valid_src = source / INNER_NEST / "valid"
    if not train_src.exists() or not valid_src.exists():
        raise FileNotFoundError(
            f"Expected {train_src} and {valid_src} to exist. "
            f"Did you point --source at the right place?"
        )

    train_dst = output / INNER_NEST / "train"
    valid_dst = output / INNER_NEST / "valid"

    all_classes = sorted(p.name for p in train_src.iterdir() if p.is_dir())
    chosen = rng.sample(all_classes, k=min(classes_per_subset, len(all_classes)))
    print(f"Selected {len(chosen)} classes: {chosen}")

    for class_name in chosen:
        _copy_sample(train_src / class_name, train_dst / class_name, train_per_class, rng)
        _copy_sample(valid_src / class_name, valid_dst / class_name, val_per_class, rng)

    train_total = _count_images(train_dst)
    valid_total = _count_images(valid_dst)
    print(f"Subset created at {output}")
    print(f"  train: {train_total} images")
    print(f"  valid: {valid_total} images")


def _copy_sample(src: Path, dst: Path, k: int, rng: random.Random) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src.iterdir() if p.is_file())
    sample = rng.sample(images, k=min(k, len(images)))
    for img in sample:
        shutil.copy2(img, dst / img.name)


def _count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


if __name__ == "__main__":
    fire.Fire(make_subset)
