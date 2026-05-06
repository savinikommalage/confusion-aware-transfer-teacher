# Transfer Teacher Curriculum Training Framework

Compact research code for CIFAR-10 curriculum, anticurriculum, and standard training.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

1. Download [CIFAR-10 (Python version)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Extract into `dataset/`:
   ```bash
   cd dataset
   tar -xvzf cifar-10-python.tar.gz
   ```
3. Convert to PNG:
   ```bash
   python convert_cifar10.py
   ```

This produces `dataset/train/` (50k images) and `dataset/test/` (10k images) with filenames like `00000_6_frog.png`.

Our method uses index matching, so for absolute correctness the labels are embedded directly in the filenames of the raw dataset images.

## Pipeline

1. **Score difficulty** — trains a teacher model and saves a checkpoint:
   ```bash
   python scripts/scoring.py --model-name resnet18 --seed 42 --epochs 30
   ```

2. **Evaluate** — runs the saved checkpoint over the full train & test sets to produce difficulty-ordered CSVs:
   ```bash
   python scripts/evaluate.py "scripts/checkpoint/<run_dir>/ckpt.pth" both --model-name ResNet18
   ```
   This saves two files to `scripts/csv/`:
   - `<ckpt_name>_difficulty_ordered_train.csv`
   - `<ckpt_name>_difficulty_ordered_test.csv`

3. **Train** (choose `--mode`: `curriculum`, `anticurriculum`, or `standard`):
   ```bash
   python scripts/train.py --mode curriculum --model-name ResNet18 --epochs 100 \
     --difficulty-train-csv csv/<ckpt_name>_difficulty_ordered_train.csv \
     --difficulty-test-csv csv/<ckpt_name>_difficulty_ordered_test.csv
   ```

## Key Arguments

### `train.py`

| Argument | Options / Default |
|---|---|
| `--mode` | `standard`, `curriculum`, `anticurriculum` |
| `--model-name` | `cnn`, `resnet18`, `vgg16`, `wideresnet` |
| `--epochs`, `--batch-size`, `--seed` | training config |
| `--train-dir`, `--test-dir` | image directories (default: `dataset/train`, `dataset/test`) |
| `--difficulty-train-csv` | path to difficulty-ordered train CSV |
| `--difficulty-test-csv` | path to difficulty-ordered test CSV |
| `--random-crop`, `--random-horizontal-flip` | augmentation flags |

### `evaluate.py`

| Argument | Options / Default |
|---|---|
| `checkpoint` (positional) | path to `.pth` checkpoint file |
| `dataset` (positional) | `train`, `test`, `both` (default: `both`) |
| `--model-name` | `ResNet18` (default) |
| `--output-dir` | output directory for CSVs (default: `csv`) |
| `--batch-size` | `128` (default) |

## Output

Training writes timestamped folders under `logs/`, `plots/`, and `checkpoint/`.
