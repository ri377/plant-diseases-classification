"""Main training entry point invoked from `commands.py train`."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from plant_diseases_classification.data.datamodule import PlantDiseasesDataModule
from plant_diseases_classification.data.download import ensure_dataset
from plant_diseases_classification.models.module import PlantDiseasesClassifier
from plant_diseases_classification.training.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)
from plant_diseases_classification.utils.logging import build_loggers
from plant_diseases_classification.utils.seed import set_seed

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _train(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # 1. Data
    data_root = ensure_dataset(
        data_root=Path(cfg.data.root),
        source=cfg.data.source,
        dvc_target=cfg.data.dvc_target,
    )
    datamodule = PlantDiseasesDataModule(
        data_root=data_root,
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        val_test_split_seed=cfg.seed,
        class_mapping_path=Path(cfg.artifacts_dir) / "class_mapping.json",
    )
    datamodule.setup()

    steps_per_epoch = len(datamodule.train_dataloader())
    effective_epochs = cfg.training.max_epochs

    # 2. Model
    model = PlantDiseasesClassifier(
        num_classes=datamodule.num_classes,
        max_lr=cfg.training.max_lr,
        weight_decay=cfg.training.weight_decay,
        total_epochs=effective_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # 3. Loggers and callbacks
    log_dir = Path(cfg.logs_dir)
    loggers = build_loggers(cfg, log_dir=log_dir)
    csv_logger = next(lg for lg in loggers if lg.__class__.__name__ == "CSVLogger")

    checkpoints_dir = Path(cfg.checkpoints_dir)
    callbacks = [
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_acc:.4f}",
            dirpath=str(checkpoints_dir),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # 4. Trainer
    trainer_kwargs = dict(
        max_epochs=effective_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    if cfg.training.limit_train_batches != 1.0:
        trainer_kwargs["limit_train_batches"] = cfg.training.limit_train_batches
    if cfg.training.limit_val_batches != 1.0:
        trainer_kwargs["limit_val_batches"] = cfg.training.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    # 5. Test on hold-out
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    log.info("Test results: %s", test_results)

    # 6. Plots
    plots_dir = Path(cfg.plots_dir)
    plot_training_curves(
        metrics_csv=Path(csv_logger.log_dir) / "metrics.csv",
        output_path=plots_dir / "training_curves.png",
    )
    confusion = model.test_confusion.compute().cpu().numpy()
    plot_confusion_matrix(
        matrix=confusion,
        class_names=datamodule.classes,
        output_path=plots_dir / "confusion_matrix.png",
    )

    log.info("Training complete. Best checkpoint: %s", callbacks[0].best_model_path)


def run_training(*overrides: str) -> None:
    """CLI entry point invoked by `commands.py train`.

    Accepts Hydra-style overrides as keyword args, e.g.
        python commands.py train training=smoke_test training.batch_size=16
    """
    # Build Hydra overrides from kwargs
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        cfg = compose(config_name="config", overrides=list(overrides))
    _train(cfg)


if __name__ == "__main__":
    # Allow running via Hydra decorator too, for `python -m plant_diseases_classification.training.train`
    @hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
    def main(cfg: DictConfig) -> None:
        _train(cfg)

    main()
