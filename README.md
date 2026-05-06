# Transfer Teacher Curriculum Training Framework 

Compact research code for CIFAR-10 curriculum, anticurriculum, and standard training.

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Sanity check

```bash
python -m py_compile model_factory.py train.py evaluate.py scoring.py models/__init__.py
```

## Pipeline order

1. Generate difficulty-ranked CSVs:

```bash
python scoring.py --model-name resnet18 --seed 42 --epochs 30
```

2. Train with curriculum learning:

```bash
python train.py --mode curriculum --model-name resnet18 --epochs 100 \
  --difficulty-train-csv csv/resnet18_cifar10_100pct_seed42_20260104_100055_difficulty_ordered_train.csv \
  --difficulty-test-csv csv/resnet18_cifar10_095243_difficulty_ordered_test.csv
```

3. Train with anticurriculum learning:

```bash
python train.py --mode anticurriculum --model-name resnet18 --epochs 100 \
  --difficulty-train-csv csv/resnet18_cifar10_100pct_seed42_20260104_100055_difficulty_ordered_train.csv \
  --difficulty-test-csv csv/resnet18_cifar10_095243_difficulty_ordered_test.csv
```

4. Standard baseline:

```bash
python train.py --mode standard --model-name resnet18 --epochs 100
```

## Key arguments

- `--mode`: `standard`, `curriculum`, or `anticurriculum`
- `--model-name`: `cnn`, `resnet18`, `vgg16`, `wideresnet`
- `--epochs`, `--batch-size`, `--num-workers`, `--seed`
- `--difficulty-train-csv`, `--difficulty-test-csv`
- `--train-dir`, `--test-dir`
- `--mapping-train-csv`, `--mapping-test-csv`
- `--random-crop`, `--random-horizontal-flip`
- `--log-dir`, `--plot-dir`, `--checkpoint-dir`, `--run-name`



## Example full command

```bash
python train.py --mode curriculum --model-name resnet18 --epochs 100 \
  --difficulty-train-csv csv/resnet18_cifar10_100pct_seed42_20260104_100055_difficulty_ordered_train.csv \
  --difficulty-test-csv csv/resnet18_cifar10_095243_difficulty_ordered_test.csv \
  --random-crop --random-horizontal-flip
```

## Output

Training writes timestamped folders under `logs/`, `plots/`, and `checkpoint/` containing:

- `training.log`
- `args.json`
- plots
- checkpoints
