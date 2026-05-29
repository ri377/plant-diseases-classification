# Plant Diseases Classification

Multi-class classification of plant leaf diseases (38 classes across 14 crops) with PyTorch Lightning.

Dataset: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

## Project layout

```
plant_diseases_classification/   # Python package
  data/        - LightningDataModule, transforms, download helper
  models/      - CNN architecture + LightningModule
  training/    - training entry point + plotting
  inference/   - inference entry point
  utils/       - logging, seeding
configs/       - Hydra configs (hierarchical)
notebooks/     - Colab/Jupyter notebooks
commands.py    - CLI dispatcher (fire)
infer.py       - thin inference wrapper
```

## Setup

Install [uv](https://docs.astral.sh/uv/) first, then:

```bash
uv sync
uv run pre-commit install
```

## Data

Data is tracked with DVC.

```bash
dvc pull -r data-storage
```

Alternatively, download from Kaggle directly into `data/`:

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data --unzip
```

## Train

Smoke test on CPU (1% of data, 1 epoch — verifies the pipeline):

```bash
uv run python commands.py train training=smoke_test
```

Full training (GPU recommended):

```bash
uv run python commands.py train
```

Override anything from the CLI thanks to Hydra:

```bash
uv run python commands.py train training.max_epochs=10 training.batch_size=64
```

## Inference

```bash
uv run python infer.py --image_path path/to/leaf.jpg --checkpoint checkpoints/best.ckpt
```

## Run on Google Colab

See `notebooks/colab-training-runner.ipynb` — it clones this repo, installs deps and runs full training on a free Colab GPU.

## MLflow

A local MLflow tracking server is expected at `http://127.0.0.1:8080`. Start it with:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Set `logger.use_mlflow=false` to skip MLflow (only CSV logging).
