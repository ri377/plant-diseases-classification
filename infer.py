"""Thin wrapper for inference. Equivalent to `python commands.py infer ...`."""

from __future__ import annotations

import fire

from plant_diseases_classification.inference.predict import run_inference

if __name__ == "__main__":
    fire.Fire(run_inference)
