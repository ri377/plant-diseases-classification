"""Import a Lightning CSVLogger metrics.csv into a local MLflow tracking server.

Reads the CSV produced by CSVLogger (logs/<name>/version_X/metrics.csv) and
recreates it as a single MLflow run so you can view the Colab training run in your
local MLflow UI alongside other runs.

Usage:
    python scripts/import_csv_to_mlflow.py \
        --csv_path "path/to/metrics.csv" \
        --tracking_uri http://127.0.0.1:8080 \
        --experiment_name plant-diseases-classification \
        --run_name colab-full-training

Optionally point --hparams at the CSVLogger hparams.yaml to also log hyperparameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import fire
import pandas as pd


def import_csv(
    csv_path: str,
    tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "plant-diseases-classification",
    run_name: str = "colab-full-training",
    hparams: str | None = None,
) -> None:
    import mlflow

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"Expected a 'step' column in {csv_path}; found {list(df.columns)}")

    metric_columns = [c for c in df.columns if c not in ("step", "epoch")]
    print(f"Found metric columns: {metric_columns}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Optional hyperparameters from a CSVLogger hparams.yaml
        if hparams:
            _log_hparams(mlflow, Path(hparams))

        mlflow.set_tag("source", "imported_from_csv")
        mlflow.set_tag("original_csv", str(csv_path))

        logged = 0
        for _, row in df.iterrows():
            step = int(row["step"]) if not math.isnan(row["step"]) else 0
            for col in metric_columns:
                value = row[col]
                if pd.notna(value):
                    mlflow.log_metric(col, float(value), step=step)
                    logged += 1

        print(f"Logged {logged} metric points across {len(df)} rows.")
        print(f"Run '{run_name}' created in experiment '{experiment_name}'.")
        print(f"Open {tracking_uri} to view it.")


def _log_hparams(mlflow_module, hparams_path: Path) -> None:
    if not hparams_path.exists():
        print(f"hparams file not found at {hparams_path}, skipping.")
        return
    try:
        import yaml

        with open(hparams_path, encoding="utf-8") as f:
            params = yaml.safe_load(f)
        if isinstance(params, dict):
            # Flatten one level for nested dicts
            flat = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}.{sub_k}"] = sub_v
                else:
                    flat[k] = v
            mlflow_module.log_params(flat)
            print(f"Logged {len(flat)} hyperparameters from {hparams_path}")
    except Exception as exc:
        print(f"Could not parse hparams ({exc}); skipping.")


if __name__ == "__main__":
    fire.Fire(import_csv)
