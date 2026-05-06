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

1. **Score difficulty:**
   ```bash
   python scoring.py --model-name resnet18 --seed 42 --epochs 30
   ```

2. **Train** (choose `--mode`: `curriculum`, `anticurriculum`, or `standard`):
   ```bash
   python train.py --mode curriculum --model-name resnet18 --epochs 100 \
     --difficulty-train-csv csv/<train_csv> \
     --difficulty-test-csv csv/<test_csv>
   ```

## Key Arguments

| Argument | Options |
|---|---|
| `--mode` | `standard`, `curriculum`, `anticurriculum` |
| `--model-name` | `cnn`, `resnet18`, `vgg16`, `wideresnet` |
| `--epochs`, `--batch-size`, `--seed` | training config |
| `--train-dir`, `--test-dir` | image directories (default: `dataset/train`, `dataset/test`) |
| `--random-crop`, `--random-horizontal-flip` | augmentation flags |

## Output

Training writes timestamped folders under `logs/`, `plots/`, and `checkpoint/`.
