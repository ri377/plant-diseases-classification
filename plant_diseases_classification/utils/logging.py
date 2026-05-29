"""Build PyTorch Lightning loggers (MLflow + CSV)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_loggers(cfg: DictConfig, log_dir: Path) -> list[Logger]:
    loggers: list[Logger] = [CSVLogger(save_dir=str(log_dir), name=cfg.logger.csv_experiment_name)]
    if cfg.logger.use_mlflow:
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.logger.mlflow_experiment_name,
            tracking_uri=cfg.logger.mlflow_tracking_uri,
            tags={"git_commit": get_git_commit(), "model": cfg.model.name},
        )
        mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(mlflow_logger)
    else:
        log.info("MLflow logging is disabled (logger.use_mlflow=false).")
    return loggers
