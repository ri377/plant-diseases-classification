"""Inference on a single image or a folder of images using a trained checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image

from plant_diseases_classification.data.transforms import build_preprocessing
from plant_diseases_classification.models.module import PlantDiseasesClassifier

log = logging.getLogger(__name__)


def _load_class_mapping(class_mapping_path: Path) -> dict[int, str]:
    with open(class_mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def _iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for suffix in (".jpg", ".jpeg", ".png"):
        yield from path.rglob(f"*{suffix}")


def run_inference(
    image_path: str,
    checkpoint: str,
    class_mapping: str = "artifacts/class_mapping.json",
    img_size: int = 224,
    device: str | None = None,
) -> list[dict]:
    """Predict class for one image or all images under a folder.

    Returns a list of {path, predicted_class, confidence} dicts and prints them as JSON.
    """
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint)
    class_mapping_path = Path(class_mapping)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mapping = _load_class_mapping(class_mapping_path)
    preprocessing = build_preprocessing(img_size)

    model = PlantDiseasesClassifier.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.eval()
    model.to(device)

    results: list[dict] = []
    with torch.no_grad():
        for img_path in _iter_images(image_path):
            image = Image.open(img_path).convert("RGB")
            tensor = preprocessing(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            results.append(
                {
                    "path": str(img_path),
                    "predicted_class": mapping[pred_idx.item()],
                    "confidence": float(conf.item()),
                }
            )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results
