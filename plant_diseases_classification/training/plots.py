"""Plotting helpers used at the end of training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_curves(metrics_csv: Path, output_path: Path) -> None:
    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        return

    epoch_metrics = (
        df.groupby("epoch")
        .agg(
            train_loss=("train_loss", "max") if "train_loss" in df.columns else ("epoch", "max"),
            train_acc=("train_acc", "max") if "train_acc" in df.columns else ("epoch", "max"),
            val_loss=("val_loss", "max") if "val_loss" in df.columns else ("epoch", "max"),
            val_acc=("val_acc", "max") if "val_acc" in df.columns else ("epoch", "max"),
            val_f1=("val_f1", "max") if "val_f1" in df.columns else ("epoch", "max"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "train_loss" in df.columns and "val_loss" in df.columns:
        axes[0].plot(epoch_metrics["epoch"], epoch_metrics["train_loss"], "o-", label="train")
        axes[0].plot(epoch_metrics["epoch"], epoch_metrics["val_loss"], "o-", label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    for col, label in [("train_acc", "train_acc"), ("val_acc", "val_acc"), ("val_f1", "val_f1")]:
        if col in df.columns:
            axes[1].plot(epoch_metrics["epoch"], epoch_metrics[col], "o-", label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        matrix,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Greens",
        cbar=True,
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix on hold-out test")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
