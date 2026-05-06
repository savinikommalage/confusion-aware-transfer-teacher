#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the full train and/or test set and generate
difficulty-ordered CSV files for curriculum learning.

The script is self-contained and does not depend on config.py or helper modules
outside this repository's core model definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model_factory import create_model


DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_STD = (0.2023, 0.1994, 0.2010)


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


def resolve_repo_path(path_value: str) -> str:
    base_dir = Path(__file__).resolve().parent
    path = Path(path_value)
    return str(path if path.is_absolute() else base_dir / path)


def compute_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class CifarMappingDataset(Dataset):
    def __init__(self, mapping_csv: str, images_dir: str, transform=None):
        self.mapping = pd.read_csv(mapping_csv)
        self.images_dir = images_dir
        self.transform = transform

        required = {"filename", "class_id"}
        missing = required - set(self.mapping.columns)
        if missing:
            raise ValueError(f"Mapping CSV is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        row = self.mapping.iloc[index]
        image_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["class_id"])
        filename = str(row["filename"])
        md5 = str(row["md5"]) if "md5" in row and pd.notna(row["md5"]) else compute_md5(image_path)
        sample_index = int(row["index"]) if "index" in row and pd.notna(row["index"]) else int(index)

        return image, label, sample_index, filename, md5


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def confusion_var_normalized(prob_vec: np.ndarray, true_label: int) -> float:
    p_true = float(prob_vec[true_label])
    mass = 1.0 - p_true
    if mass < 1e-12:
        return 0.0
    false_probs = np.delete(prob_vec, true_label) / mass
    return float(np.var(false_probs))


def evaluate_dataset(model, mapping_csv, images_dir, device, batch_size, num_classes):
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    dataset = CifarMappingDataset(mapping_csv, images_dir, transform=get_eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    total_batches = len(loader)
    print(f"  Processing {total_batches} batches...")

    with torch.no_grad():
        for batch_idx, (images, true_y, sample_index, filename, md5) in enumerate(loader):
            images = images.to(device)
            true_y = true_y.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                pct = ((batch_idx + 1) / total_batches) * 100.0
                print(f"    Batch {batch_idx + 1}/{total_batches} ({pct:.1f}%)")

            pred_prob, pred_y = probs.max(dim=1)
            true_prob = probs[torch.arange(true_y.size(0), device=device), true_y]

            probs_np = probs.detach().cpu().numpy()
            true_np = true_y.detach().cpu().numpy()
            pred_np = pred_y.detach().cpu().numpy()
            predp_np = pred_prob.detach().cpu().numpy()
            truep_np = true_prob.detach().cpu().numpy()
            index_np = sample_index.detach().cpu().numpy()
            filename_list = [str(item) for item in filename]
            md5_list = [str(item) for item in md5]

            for sample_idx in range(len(true_np)):
                target = int(true_np[sample_idx])
                probability_vector = probs_np[sample_idx]
                confusion_var = confusion_var_normalized(probability_vector, target)
                probability_true = float(truep_np[sample_idx])
                difficulty = (1.0 - probability_true) * confusion_var

                record = {
                    "index": int(index_np[sample_idx]),
                    "filename": filename_list[sample_idx],
                    "md5": md5_list[sample_idx],
                    "true_label": target,
                    "pred_label": int(pred_np[sample_idx]),
                    "prob_pred": float(predp_np[sample_idx]),
                    "prob_true": probability_true,
                    "confusion_var": confusion_var,
                    "difficulty": float(difficulty),
                }
                for class_idx in range(num_classes):
                    record[f"prob_class_{class_idx}"] = float(probability_vector[class_idx])
                rows.append(record)

    df = pd.DataFrame(rows)
    df = df.sort_values(["difficulty", "index"], ascending=[True, True]).reset_index(drop=True)
    return df


def build_parser():
    parser = argparse.ArgumentParser(description="Generate difficulty CSVs from a checkpoint")
    parser.add_argument("checkpoint", help="Path to the best checkpoint file")
    parser.add_argument("dataset", nargs="?", default="both", choices=["train", "test", "both"], help="Which split to evaluate")
    parser.add_argument("--model-name", default="ResNet18", help="Model architecture name")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mapping-train-csv", default="csv/mapping_train.csv")
    parser.add_argument("--mapping-test-csv", default="csv/mapping_test.csv")
    parser.add_argument("--train-images-dir", default="dataset/train")
    parser.add_argument("--test-images-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="csv")
    return parser


def main():
    args = build_parser().parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mapping_train_csv = resolve_repo_path(args.mapping_train_csv)
    mapping_test_csv = resolve_repo_path(args.mapping_test_csv)
    train_images_dir = resolve_repo_path(args.train_images_dir)
    test_images_dir = resolve_repo_path(args.test_images_dir)
    output_dir = resolve_repo_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("EVALUATING CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    canonical_model_name = normalize_model_name(args.model_name)
    print(f"Model: {canonical_model_name}")
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    model = create_model(canonical_model_name, model_args={"num_classes": args.num_classes}).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    model.load_state_dict(state)
    model.eval()

    ckpt_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if ckpt_basename.endswith("_best"):
        ckpt_basename = ckpt_basename[:-5]

    if args.dataset in ["train", "both"]:
        print(f"Evaluating TRAIN set ({mapping_train_csv})...")
        df_train = evaluate_dataset(
            model,
            mapping_train_csv,
            train_images_dir,
            device,
            args.batch_size,
            args.num_classes,
        )
        train_out = os.path.join(output_dir, f"{ckpt_basename}_difficulty_ordered_train.csv")
        df_train.to_csv(train_out, index=False)
        print(f"Saved: {os.path.abspath(train_out)}")
        print(f"  Rows: {len(df_train)} | Columns: {len(df_train.columns)}\n")

    if args.dataset in ["test", "both"]:
        print(f"Evaluating TEST set ({mapping_test_csv})...")
        df_test = evaluate_dataset(
            model,
            mapping_test_csv,
            test_images_dir,
            device,
            args.batch_size,
            args.num_classes,
        )
        test_out = os.path.join(output_dir, f"{ckpt_basename}_difficulty_ordered_test.csv")
        df_test.to_csv(test_out, index=False)
        print(f"Saved: {os.path.abspath(test_out)}")
        print(f"  Rows: {len(df_test)} | Columns: {len(df_test.columns)}\n")

    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
