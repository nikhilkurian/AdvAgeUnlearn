# Medical Image Classification – Minimal Run Guide

This repository contains a PyTorch Lightning pipeline to train and evaluate multilabel chest X-ray classifiers (14 classes). This document includes only what you need to install, configure, and run.

## 1) Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- Install Python packages:

```bash
pip install -r requirements.txt
```

## 2) Data Expectations

You need two kinds of CSV files:

1) Dataset CSVs (for train/val/test), e.g. `train_mimic.csv`:
- Must include column `path_preproc` (relative path to preprocessed image)
- Must include 14 label columns with 0/1 values:
  - No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- Optional columns: `age` or `AgeGroup` (if `age` exists, age groups can be inferred)

2) Metadata CSVs (for train/val/test), e.g. `train_final.csv`:
- Must include column `img_path` (absolute path to the image)
- Can include any additional columns (StudyDate, Age, Sex, View, etc.) for merging into outputs

Note on paths: if the dataset images and metadata `img_path` use different root prefixes, the training script remaps dataset paths to match metadata before merging predictions/embeddings.

## 3) Configure

All settings are in `config.yaml`. Update the paths and essential parameters.

Minimal set of keys to edit:

```yaml
# Outputs
out_name_prefix: 'experiment'
out_base_dir: './outputs'
log_dir: './logs'
checkpoint_dir: './checkpoints'

# Data paths (edit all)
train_csv: '/absolute/path/to/train_mimic.csv'
val_csv: '/absolute/path/to/val_mimic.csv'
test_csv: '/absolute/path/to/test_mimic.csv'
train_details_csv: '/absolute/path/to/train_final.csv'
val_details_csv: '/absolute/path/to/val_final.csv'
test_details_csv: '/absolute/path/to/test_final.csv'
img_data_dir: '/absolute/path/to/images_root/'

# Model (choose one model_type)
model_type: 'DenseNetAgeAdv'   # Options: DenseNet, ResNet, VisionTransformer, DenseNetAgeAdv
num_classes_main: 14
adv_age_lambda: 0.1            # used only by DenseNetAgeAdv

# Data loading
image_size: [224, 224]
batch_size_main: 150
num_workers: 4

# Training
epochs_main: 1                 # increase for real runs
enable_early_stopping: True
early_stopping_patience: 10

# Age groups (used for metadata-aware training/eval)
age_groups:
  - [0, 36, "0-36"]
  - [36, 50, "36-50"]
  - [50, 65, "50-65"]
  - [65, -1, "65+"]
```

## 4) Run

Train + validate (and test + export predictions/embeddings after training):

```bash
python train.py --config config.yaml --dev 0
```

Resume from a checkpoint:

```bash
python train.py --config config.yaml --checkpoint_path /path/to/checkpoint.ckpt --dev 0
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

## 5) Outputs

For each run, files are written under the output directory specified in `config.yaml`:

- predictions.train.epoch_X.csv
- predictions.val.model_<id>.csv
- predictions.test.model_<id>.csv
- embeddings.train.epoch_X.csv
- embeddings.val.model_<id>.csv
- embeddings.test.model_<id>.csv
- per_class_auc.txt (per-class and macro AUC)
- checkpoints directory with best/last checkpoints

Each predictions CSV contains: per-class probabilities, logits, targets, `img_path`, and merged metadata. Each embeddings CSV contains a feature vector per image plus targets and metadata.

## 6) Common Issues

- Empty CSVs after run: ensure `img_path` in metadata matches dataset image paths (root prefixes must align). The script remaps common roots, but custom setups may require adjusting paths or metadata.
- CUDA out of memory: reduce `batch_size_main` or `image_size`.
- Slow loading: increase `num_workers`.
- File not found: verify all paths in `config.yaml` are absolute and readable. 