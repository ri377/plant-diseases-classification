# Plant Diseases Classification

Многоклассовая классификация болезней листьев сельскохозяйственных культур (38 классов, 14 видов растений) на основе свёрточной нейросети с residual-связями, обёрнутой в PyTorch Lightning.

Проект выполнен в рамках курса MLOps и реализует полный цикл: подготовку данных, обучение, логирование экспериментов, версионирование данных и моделей, упаковку и инференс.

Датасет: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

## Содержание

- [Постановка задачи](#постановка-задачи)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Данные](#данные)
- [Обучение](#обучение)
- [Логирование экспериментов](#логирование-экспериментов)
- [Результаты](#результаты)
- [Инференс](#инференс)
- [Обучение в Google Colab](#обучение-в-google-colab)
- [Используемые технологии](#используемые-технологии)

## Постановка задачи

По фотографии листа растения модель определяет вид культуры и её состояние — здоров лист или поражён конкретным заболеванием. Всего 38 классов (комбинации «культура + болезнь», например `Tomato___Late_blight` или `Apple___healthy`).

Практическая ценность: автоматический помощник для агрономов и фермеров, позволяющий на ранней стадии выявлять болезни растений без лабораторной диагностики.

**Формат данных:**

- Вход: RGB-изображение, приводится к тензору `(3, 224, 224)` с нормализацией по статистикам ImageNet.
- Выход: вектор логитов `(38,)`; после softmax — распределение вероятностей по классам. Индекс класса отображается в читаемое имя через `class_mapping.json`.

**Метрики:** основная — accuracy; дополнительные — macro-F1 (для контроля слабых классов) и confusion matrix (для анализа путаемых болезней).

## Структура проекта

```
plant-diseases-classification/
├── plant_diseases_classification/   # Python-пакет с основной логикой
│   ├── data/
│   │   ├── datamodule.py            # LightningDataModule + stratified split
│   │   ├── transforms.py            # препроцессинг (resize, normalize)
│   │   └── download.py              # загрузка данных (dvc / kaggle / local)
│   ├── models/
│   │   ├── cnn.py                   # архитектура PlantDiseaseCNN
│   │   └── module.py                # LightningModule с метриками
│   ├── training/
│   │   ├── train.py                 # точка входа обучения
│   │   └── plots.py                 # построение графиков и confusion matrix
│   ├── inference/
│   │   └── predict.py               # инференс на изображении или папке
│   └── utils/
│       ├── logging.py               # настройка MLflow + CSV логгеров
│       └── seed.py                  # фиксация seed
├── configs/                         # иерархические конфиги Hydra
│   ├── config.yaml                  # корневой конфиг
│   ├── data/plant_village.yaml
│   ├── model/baseline_cnn.yaml
│   ├── training/default.yaml
│   ├── training/smoke_test.yaml     # быстрый прогон для отладки
│   └── logger/default.yaml
├── scripts/
│   ├── make_subset.py               # создание небольшого сабсета для отладки
│   └── import_csv_to_mlflow.py      # импорт CSV-метрик в MLflow
├── notebooks/
│   └── colab-training-runner.ipynb  # запуск обучения в Google Colab
├── commands.py                      # единая точка входа CLI (train / infer)
├── infer.py                         # обёртка для инференса
├── pyproject.toml                   # зависимости (uv)
├── .pre-commit-config.yaml          # хуки качества кода
└── README.md
```

## Установка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.

```bash
git clone https://github.com/ri377/plant-diseases-classification.git
cd plant-diseases-classification

uv sync
uv run pre-commit install
```

## Данные

Данные версионируются через DVC. Способ получения данных задаётся в `configs/data/plant_village.yaml` параметром `source` (`local`, `dvc` или `kaggle`).

**Получить данные из DVC remote:**

```bash
uv run dvc pull -r data-storage
```

**Скачать напрямую с Kaggle** (требуется настроенный Kaggle API token):

```bash
uv run kaggle datasets download -d vipoooool/new-plant-diseases-dataset \
    -p data/new-plant-diseases-dataset --unzip
```

**Создать небольшой сабсет для локальной отладки** (около 50 МБ вместо 3 ГБ):

```bash
uv run python scripts/make_subset.py \
    --source data/new-plant-diseases-dataset \
    --output data/new-plant-diseases-dataset-subset
```

Скрипт автоматически находит папки `train/` и `valid/` на любой глубине вложенности и копирует сбалансированный сабсет.

### DVC remotes

В проекте настроено два DVC remote (для данных и для моделей):

```bash
# Просмотр настроенных remote
uv run dvc remote list
```

Чувствительные параметры (пароли, ключи доступа) хранятся в `.dvc/config.local`, который исключён из git.

## Обучение

Все параметры обучения управляются через Hydra и могут переопределяться из командной строки.

**Быстрый прогон для проверки пайплайна** (1 эпоха, 1% данных):

```bash
uv run python commands.py train training=smoke_test
```

**Полное обучение:**

```bash
uv run python commands.py train
```

**Переопределение любых параметров:**

```bash
uv run python commands.py train training.max_epochs=10 training.batch_size=64
```

После обучения создаются:

- `checkpoints/` — лучший чекпойнт по `val_acc` (сохраняется через `ModelCheckpoint`);
- `plots/training_curves.png` — кривые loss и метрик;
- `plots/confusion_matrix.png` — матрица ошибок на отложенной тестовой выборке;
- `artifacts/class_mapping.json` — соответствие индексов классов их именам;
- `logs/` — CSV-логи метрик.

### Разделение выборок

Датасет поставляется с готовым разбиением `train` / `valid`. Валидационная часть дополнительно делится пополам (со стратификацией по классам) на validation и отложенный hold-out test. Воспроизводимость обеспечивается фиксацией seed и детерминированным режимом Lightning.

## Логирование экспериментов

Используются два логгера одновременно: **MLflow** (метрики, гиперпараметры, git commit) и **CSV** (для локального построения графиков).

Запуск локального MLflow-сервера:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Обучение с включённым MLflow:

```bash
uv run python commands.py train logger.use_mlflow=true
```

Интерфейс доступен по адресу `http://127.0.0.1:8080`. Логируются 9+ метрик (train/val/test для loss, accuracy, F1), learning rate, гиперпараметры и тег `git_commit` для привязки прогона к версии кода.

Если обучение проводилось в Colab, метрики из `metrics.csv` можно импортировать в локальный MLflow:

```bash
uv run python scripts/import_csv_to_mlflow.py \
    --csv_path "logs/plant_diseases/version_0/metrics.csv" \
    --run_name colab-full-training
```

## Результаты

Полноценное обучение (полный датасет, обучение на GPU в Google Colab):

| Метрика    | Значение |
| ---------- | -------- |
| test_acc   | 0.967    |
| test_f1    | 0.967    |
| val_acc    | 0.970    |
| val_f1     | 0.970    |
| val_loss   | 0.101    |
| train_acc  | 0.957    |
| train_loss | 0.150    |

Модель уверенно достигает целевого качества (accuracy > 0.95). Кривые обучения показывают стабильную сходимость без признаков переобучения: train и validation метрики растут согласованно.

<img width="947" height="523" alt="Screenshot 2026-05-30 125412" src="https://github.com/user-attachments/assets/aa6893fe-372f-4715-be25-a8fcb76297c1" />

<img width="1648" height="712" alt="Screenshot 2026-05-30 125447" src="https://github.com/user-attachments/assets/5eea8cd7-b5db-46b4-a433-9011414a441d" />


## Инференс

```bash
uv run python infer.py \
    --image_path path/to/leaf.jpg \
    --checkpoint checkpoints/best.ckpt \
    --class_mapping artifacts/class_mapping.json
```

Принимает путь к одному изображению или к папке с изображениями. Возвращает JSON со списком предсказаний: путь, предсказанный класс и уверенность модели.

## Обучение в Google Colab

Для обучения на бесплатном GPU используется ноутбук `notebooks/colab-training-runner.ipynb`. Он клонирует репозиторий, устанавливает зависимости, скачивает датасет с Kaggle и запускает полное обучение. Результаты (чекпойнты, графики, логи) сохраняются на Google Drive.

Краткая последовательность:

1. Открыть ноутбук в Colab, выбрать GPU в настройках среды выполнения.
2. Загрузить Kaggle API token на Google Drive.
3. Выполнить ячейки по порядку.
4. Скачать обученный чекпойнт и метрики с Google Drive.

## Используемые технологии

- **PyTorch Lightning** — организация обучения (DataModule, LightningModule, Trainer).
- **Hydra** — иерархическая конфигурация без магических констант в коде.
- **DVC** — версионирование данных и моделей (два отдельных remote).
- **MLflow** — трекинг экспериментов, метрик и гиперпараметров.
- **torchmetrics** — вычисление accuracy, macro-F1, confusion matrix.
- **Fire** — единая точка входа CLI.
- **uv** — управление зависимостями и виртуальным окружением.
- **pre-commit + ruff** — контроль качества и форматирования кода.
