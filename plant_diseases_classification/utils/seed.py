"""Deterministic seeding utilities."""

import lightning.pytorch as pl


def set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
