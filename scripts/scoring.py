#!/usr/bin/env python3
"""
Train the teacher baseline for 30 epochs and generate difficulty CSVs.

This is a thin front-end for the pipeline:
1. Run train.py in standard mode for 30 epochs.
2. Find the best checkpoint saved by train.py.
3. Run evaluate.py on that checkpoint to produce train and test difficulty CSVs.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def resolve_repo_path(path_value: str) -> str:
    base_dir = Path(__file__).resolve().parent
    path = Path(path_value)
    return str(path if path.is_absolute() else base_dir / path)


def normalize_model_name(model_name: str) -> str:
    aliases = {
        "cnn": "CNN",
        "resnet18": "ResNet18",
        "vgg16": "VGG16",
        "vgg": "VGG16",
        "wide-resnet": "WideResNet",
        "wideresnet": "WideResNet",
    }
    key = model_name.strip()
    return aliases.get(key.lower(), key)


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare teacher checkpoint and difficulty CSVs")
    parser.add_argument("--model-name", default="ResNet18", help="Model architecture for the teacher baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30, help="Teacher training epochs; default is 30")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-dir", default="cifar10_raw/train")
    parser.add_argument("--test-dir", default="cifar10_raw/test")
    parser.add_argument("--mapping-train-csv", default="csv/mapping_train.csv")
    parser.add_argument("--mapping-test-csv", default="csv/mapping_test.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--output-dir", default="csv")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--evaluate-script", default="evaluate.py")
    return parser


def find_latest_best_checkpoint(checkpoint_dir: str, run_prefix: str) -> str:
    pattern = os.path.join(checkpoint_dir, f"{run_prefix}_*", "ckpt.pth")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found with pattern: {pattern}")
    return matches[-1]


def run_command(command):
    print("\nRunning:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    args = build_parser().parse_args()

    canonical_model_name = normalize_model_name(args.model_name)
    train_script = resolve_repo_path(args.train_script)
    evaluate_script = resolve_repo_path(args.evaluate_script)
    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    output_dir = resolve_repo_path(args.output_dir)
    mapping_train_csv = resolve_repo_path(args.mapping_train_csv)
    mapping_test_csv = resolve_repo_path(args.mapping_test_csv)
    train_dir = resolve_repo_path(args.train_dir)
    test_dir = resolve_repo_path(args.test_dir)

    run_prefix = f"teacher_{canonical_model_name.lower()}_{args.epochs}ep_seed{args.seed}"

    print("=" * 80)
    print("STAGE 1: TRAIN TEACHER BASELINE")
    print("=" * 80)

    train_command = [
        sys.executable,
        train_script,
        "--mode",
        "standard",
        "--model-name",
        canonical_model_name,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--train-dir",
        train_dir,
        "--test-dir",
        test_dir,
        "--mapping-train-csv",
        mapping_train_csv,
        "--mapping-test-csv",
        mapping_test_csv,
        "--checkpoint-dir",
        checkpoint_dir,
        "--run-name",
        run_prefix,
    ]
    run_command(train_command)

    best_checkpoint = find_latest_best_checkpoint(checkpoint_dir, run_prefix)
    print(f"\nBest checkpoint found: {best_checkpoint}")

    print("\n" + "=" * 80)
    print("STAGE 2: GENERATE DIFFICULTY CSVS")
    print("=" * 80)

    eval_command = [
        sys.executable,
        evaluate_script,
        best_checkpoint,
        "both",
        "--model-name",
        canonical_model_name,
        "--num-classes",
        str(args.num_classes),
        "--batch-size",
        str(args.batch_size),
        "--mapping-train-csv",
        mapping_train_csv,
        "--mapping-test-csv",
        mapping_test_csv,
        "--train-images-dir",
        train_dir,
        "--test-images-dir",
        test_dir,
        "--output-dir",
        output_dir,
    ]
    run_command(eval_command)

    ckpt_base = os.path.splitext(os.path.basename(best_checkpoint))[0]
    if ckpt_base.endswith("_best"):
        ckpt_base = ckpt_base[:-5]

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Teacher checkpoint: {best_checkpoint}")
    print(f"Train CSV: {os.path.join(output_dir, f'{ckpt_base}_difficulty_ordered_train.csv')}")
    print(f"Test CSV:  {os.path.join(output_dir, f'{ckpt_base}_difficulty_ordered_test.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
