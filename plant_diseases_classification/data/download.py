"""Data acquisition helpers.

Supports three sources, picked by config:
  - dvc:     `dvc pull` from the configured remote
  - kaggle:  Kaggle CLI download (requires kaggle.json under ~/.kaggle/)
  - local:   data already present at the configured path; verify and return
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"


def ensure_dataset(
    data_root: Path,
    source: str,
    dvc_target: str | None = None,
    kaggle_download_root: Path | str | None = None,
) -> Path:
    """Ensure dataset is present locally at `data_root`. Returns `data_root`."""
    data_root = Path(data_root)
    if data_root.exists() and any(data_root.iterdir()):
        log.info("Dataset already present at %s", data_root)
        return data_root

    if source == "dvc":
        _pull_from_dvc(dvc_target)
    elif source == "kaggle":
        if kaggle_download_root is None:
            # Fall back to project-level data/ directory.
            kaggle_download_root = Path("data") / "new-plant-diseases-dataset"
        _pull_from_kaggle(Path(kaggle_download_root))
    elif source == "local":
        raise FileNotFoundError(
            f"Data source is 'local' but {data_root} is empty. "
            "Place the dataset there manually or change `data.source` in the config."
        )
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    if not data_root.exists():
        raise FileNotFoundError(
            f"After fetching the dataset, {data_root} still does not exist. "
            "Check that `data.root` matches the directory structure produced by the source."
        )
    return data_root


def _pull_from_dvc(target: str | None) -> None:
    log.info("Pulling data via DVC (target=%s)", target)
    cmd = ["dvc", "pull", "-r", "data-storage"]
    if target:
        cmd.append(target)
    subprocess.run(cmd, check=True)


def _pull_from_kaggle(download_root: Path) -> None:
    """Download the Kaggle dataset into `download_root`.

    The Kaggle CLI extracts the dataset's own folder structure inside `download_root`.
    For this dataset the result is:
        download_root/
            New Plant Diseases Dataset(Augmented)/
                New Plant Diseases Dataset(Augmented)/
                    train/
                    valid/
            test/
    """
    log.info("Downloading dataset from Kaggle into %s", download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET_SLUG,
            "-p",
            str(download_root),
            "--unzip",
        ],
        check=True,
    )
