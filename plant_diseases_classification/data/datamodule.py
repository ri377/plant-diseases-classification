"""LightningDataModule for the New Plant Diseases Dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from plant_diseases_classification.data.transforms import build_preprocessing

log = logging.getLogger(__name__)


class PlantDiseasesDataModule(pl.LightningDataModule):
    """Loads PlantVillage-style ImageFolder data, splits valid 50/50 into val and hold-out test."""

    def __init__(
        self,
        data_root: Path,
        img_size: int,
        batch_size: int,
        num_workers: int,
        val_test_split_seed: int,
        class_mapping_path: Path | None = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_test_split_seed = val_test_split_seed
        self.class_mapping_path = (
            Path(class_mapping_path) if class_mapping_path is not None else None
        )
        self._preprocessing = build_preprocessing(img_size)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes: list[str] = []

    def setup(self, stage: str | None = None) -> None:
        train_dir = self.data_root / "train"
        valid_dir = self.data_root / "valid"
        if not train_dir.exists() or not valid_dir.exists():
            raise FileNotFoundError(
                f"Expected `train/` and `valid/` subfolders under {self.data_root}"
            )

        self.train_dataset = ImageFolder(str(train_dir), transform=self._preprocessing)
        self.classes = list(self.train_dataset.classes)

        valid_full = ImageFolder(str(valid_dir), transform=self._preprocessing)
        val_idx, test_idx = train_test_split(
            list(range(len(valid_full))),
            test_size=0.5,
            stratify=valid_full.targets,
            random_state=self.val_test_split_seed,
        )
        self.val_dataset = Subset(valid_full, val_idx)
        self.test_dataset = Subset(valid_full, test_idx)

        log.info(
            "Datasets ready: train=%d, val=%d, test=%d, num_classes=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
            len(self.classes),
        )

        if self.class_mapping_path is not None:
            self._save_class_mapping()

    def _save_class_mapping(self) -> None:
        mapping = {idx: name for idx, name in enumerate(self.classes)}
        self.class_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self.class_mapping_path.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Saved class mapping -> %s", self.class_mapping_path)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
