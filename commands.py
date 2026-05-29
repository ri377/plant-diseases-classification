"""Single CLI entry point dispatched via Fire."""

from __future__ import annotations

import fire

from plant_diseases_classification.inference.predict import run_inference
from plant_diseases_classification.training.train import run_training


def main() -> None:
    fire.Fire(
        {
            "train": run_training,
            "infer": run_inference,
        }
    )


if __name__ == "__main__":
    main()
