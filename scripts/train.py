"""Single-entry CIFAR-10 training script.

Supports three training modes:
- standard: use the full training set
- curriculum: start from the easiest samples and add harder ones over time
- anticurriculum: start from the hardest samples and add easier ones over time

All hyperparameters are provided through CLI arguments. The script writes a
timestamped run directory containing logs, plots, checkpoints, and the exact
CLI arguments used for the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from model_factory import create_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    print(" [", end="")
    print("=" * cur_len, end="")
    print(">", end="")
    print("." * rest_len, end="")
    print("]", end="")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    parts = [f"  Step: {format_time(step_time)}", f" | Tot: {format_time(tot_time)}"]
    if msg:
        parts.append(f" | {msg}")

    message = "".join(parts)
    print(message, end="")

    terminal_width = 80
    print(" " * max(0, terminal_width - int(TOTAL_BAR_LENGTH) - len(message) - 3), end="")
    print("\b" * max(0, terminal_width - int(TOTAL_BAR_LENGTH / 2) + 2), end="")
    print(f" {current + 1}/{total} ", end="")

    if current < total - 1:
        print("\r", end="")
    else:
        print()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    result = ""
    pieces = 1
    if days > 0:
        result += f"{days}D"
        pieces += 1
    if hours > 0 and pieces <= 2:
        result += f"{hours}h"
        pieces += 1
    if minutes > 0 and pieces <= 2:
        result += f"{minutes}m"
        pieces += 1
    if secondsf > 0 and pieces <= 2:
        result += f"{secondsf}s"
        pieces += 1
    if millis > 0 and pieces <= 2:
        result += f"{millis}ms"
        pieces += 1
    if result == "":
        result = "0ms"
    return result


def resolve_repo_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)


class CIFAR10RawDataset(Dataset):
    """Dataset for loading raw PNG images from the CIFAR-10 export."""

    def __init__(self, data_dir, mapping_csv, indices=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        samples = []
        with open(mapping_csv, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                samples.append(
                    {
                        "index": int(row["index"]),
                        "filename": row["filename"],
                        "class_id": int(row["class_id"]),
                        "class_name": row.get("class_name", ""),
                    }
                )

        if indices is not None:
            index_set = set(indices)
            samples = [sample for sample in samples if sample["index"] in index_set]

        self.samples = sorted(samples, key=lambda sample: sample["index"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.data_dir, sample["filename"])).convert("RGB")
        label = sample["class_id"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CIFAR10TestDataset(Dataset):
    """Test dataset sorted by batch_row so stage metrics align with positions."""

    def __init__(self, data_dir, mapping_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        samples = []
        with open(mapping_csv, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                samples.append(
                    {
                        "batch_row": int(row["batch_row"]),
                        "filename": row["filename"],
                        "class_id": int(row["class_id"]),
                        "class_name": row.get("class_name", ""),
                    }
                )

        self.samples = sorted(samples, key=lambda sample: sample["batch_row"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.data_dir, sample["filename"])).convert("RGB")
        label = sample["class_id"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def parse_int_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_difficulty_indices(difficulty_csv, num_samples, from_end=False):
    indices = []
    with open(difficulty_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            indices.append(int(row["index"]))

    if num_samples >= len(indices):
        return indices

    if from_end:
        return indices[-num_samples:]
    return indices[:num_samples]


def create_test_stages(mapping_test_csv, difficulty_test_csv, stage_sizes):
    index_to_batch_row = {}
    with open(mapping_test_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index_to_batch_row[int(row["index"])] = int(row["batch_row"])

    difficulty_indices = []
    with open(difficulty_test_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            difficulty_indices.append(int(row["index"]))

    stages = []
    start = 0
    for stage_size in stage_sizes:
        stage_indices = difficulty_indices[start : start + stage_size]
        stages.append([index_to_batch_row[index] for index in stage_indices])
        start += stage_size

    return stages


def get_gpu_memory_info(device):
    if device == "cuda" and torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}


def build_train_indices(args, epoch):
    if args.mode == "standard":
        return None

    stage = epoch // args.stage_epochs
    num_samples = min(
        args.initial_samples + stage * args.samples_per_stage,
        args.total_train_samples,
    )
    from_end = args.mode == "anticurriculum"
    return load_difficulty_indices(args.difficulty_train_csv, num_samples, from_end=from_end)


def build_train_dataset(args, indices, train_transform):
    return CIFAR10RawDataset(
        data_dir=args.train_dir,
        mapping_csv=args.mapping_train_csv,
        indices=indices,
        transform=train_transform,
    )


def make_scheduler(args, optimizer):
    if args.scheduler_type == "none":
        return None

    if args.scheduler_type == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

    if args.scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    if args.scheduler_type == "cosine_stage":
        stage_max = parse_float_list(args.stage_lr_max)
        stage_min = parse_float_list(args.stage_lr_min)
        if len(stage_max) != len(stage_min):
            raise ValueError("stage_lr_max and stage_lr_min must have the same length")

        def lr_lambda(epoch):
            stage_len = args.stage_lr_len
            stage = min(epoch // stage_len, len(stage_max) - 1)
            t = epoch % stage_len
            cosine = 0.5 * (1 + math.cos(math.pi * t / stage_len))
            lr = stage_min[stage] + (stage_max[stage] - stage_min[stage]) * cosine
            return lr / stage_max[0]

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.scheduler_type == "exponential_stage":
        stage_max = parse_float_list(args.stage_lr_max)
        stage_min = parse_float_list(args.stage_lr_min)
        if len(stage_max) != len(stage_min):
            raise ValueError("stage_lr_max and stage_lr_min must have the same length")

        def lr_lambda(epoch):
            stage_len = args.stage_lr_len
            stage = min(epoch // stage_len, len(stage_max) - 1)
            t = epoch % stage_len
            decay = (stage_min[stage] / stage_max[stage]) ** (t / stage_len)
            lr = stage_max[stage] * decay
            return lr / stage_max[0]

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")


def train_epoch(model, trainloader, criterion, optimizer, device):
    start_time = time.time()
    data_loading_time = 0.0
    training_time = 0.0

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        data_loading_time += time.time() - data_start

        batch_size = targets.size(0)
        train_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        training_time += time.time() - train_start

        train_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.2f%% (%d/%d)"
            % (train_loss / total, 100.0 * correct / total, correct, total),
        )

    avg_loss = train_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    epoch_time = time.time() - start_time
    overhead_time = epoch_time - data_loading_time - training_time

    return avg_loss, accuracy, epoch_time, data_loading_time, training_time, overhead_time


def test_epoch(model, testloader, device, test_stages=None):
    start_time = time.time()
    data_loading_time = 0.0
    inference_time = 0.0

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            data_loading_time += time.time() - data_start

            batch_size = targets.size(0)
            inf_start = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            inference_time += time.time() - inf_start

            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.2f%% (%d/%d)"
                % (test_loss / total, 100.0 * correct / total, correct, total),
            )

    avg_loss = test_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    test_time = time.time() - start_time
    overhead_time = test_time - data_loading_time - inference_time

    stage_accuracies = []
    if test_stages is not None:
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        for stage_positions in test_stages:
            stage_pred = predictions[stage_positions]
            stage_targ = targets[stage_positions]
            stage_accuracies.append(100.0 * (stage_pred == stage_targ).mean())

    return avg_loss, accuracy, stage_accuracies, test_time, data_loading_time, inference_time, overhead_time


def save_plots(history, plot_dir, epoch):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(len(history["train_loss"]))

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0, 0].plot(epochs, history["test_loss"], label="Test Loss", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Test Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[0, 1].plot(epochs, history["test_acc"], label="Test Acc", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Training and Test Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if history["stage_acc"]:
        num_stages = len(history["stage_acc"][0])
        for stage_idx in range(num_stages):
            axes[0, 2].plot(
                epochs,
                [stage_acc[stage_idx] for stage_acc in history["stage_acc"]],
                label=f"L{stage_idx + 1}",
                marker="o",
            )
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Accuracy (%)")
        axes[0, 2].set_title("Stage Accuracies")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

    if history["lr"]:
        axes[1, 0].plot(epochs, history["lr"], marker="o")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].grid(True)

    if history["train_time"]:
        axes[1, 1].plot(epochs, history["train_time"], label="Train Total", marker="o")
        axes[1, 1].plot(
            epochs,
            history["train_data_loading_time"],
            label="Train Data Loading",
            marker="^",
            linestyle="--",
        )
        axes[1, 1].plot(
            epochs,
            history["train_compute_time"],
            label="Train Compute",
            marker="v",
            linestyle="--",
        )
        axes[1, 1].plot(epochs, history["test_time"], label="Test Total", marker="s")
        axes[1, 1].plot(
            epochs,
            history["test_data_loading_time"],
            label="Test Data Loading",
            marker="^",
            linestyle="--",
        )
        axes[1, 1].plot(
            epochs,
            history["test_inference_time"],
            label="Test Inference",
            marker="v",
            linestyle="--",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Time (seconds)")
        axes[1, 1].set_title("Epoch Time Breakdown")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True)

    if history["total_time"]:
        axes[1, 2].plot(epochs, [total / 60.0 for total in history["total_time"]], marker="o")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Cumulative Time (minutes)")
        axes[1, 2].set_title("Cumulative Training Time")
        axes[1, 2].grid(True)

    if history["gpu_allocated"] and any(value > 0 for value in history["gpu_allocated"]):
        axes[2, 0].plot(epochs, history["gpu_allocated"], label="Allocated", marker="o")
        axes[2, 0].plot(epochs, history["gpu_reserved"], label="Reserved", marker="s")
        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].set_ylabel("GPU Memory (GB)")
        axes[2, 0].set_title("GPU Memory Usage")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        axes[2, 1].plot(epochs, history["gpu_max_allocated"], marker="o")
        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].set_ylabel("GPU Memory (GB)")
        axes[2, 1].set_title("GPU Max Allocated Memory")
        axes[2, 1].grid(True)
    else:
        axes[2, 0].text(
            0.5,
            0.5,
            "No GPU memory data\n(CPU training)",
            ha="center",
            va="center",
            transform=axes[2, 0].transAxes,
        )
        axes[2, 0].set_title("GPU Memory Usage")
        axes[2, 1].text(
            0.5,
            0.5,
            "No GPU memory data\n(CPU training)",
            ha="center",
            va="center",
            transform=axes[2, 1].transAxes,
        )
        axes[2, 1].set_title("GPU Max Allocated Memory")

    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"training_metrics_epoch_{epoch}.png"))
    plt.close(fig)

    timing_data = {
        "epochs": list(epochs),
        "train_time": history["train_time"],
        "train_data_loading_time": history["train_data_loading_time"],
        "train_compute_time": history["train_compute_time"],
        "train_overhead_time": history["train_overhead_time"],
        "test_time": history["test_time"],
        "test_data_loading_time": history["test_data_loading_time"],
        "test_inference_time": history["test_inference_time"],
        "test_overhead_time": history["test_overhead_time"],
        "total_time": history["total_time"],
        "gpu_allocated": history["gpu_allocated"],
        "gpu_reserved": history["gpu_reserved"],
        "gpu_max_allocated": history["gpu_max_allocated"],
    }
    with open(os.path.join(plot_dir, f"timing_gpu_metrics_epoch_{epoch}.json"), "w") as handle:
        json.dump(timing_data, handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Single-file CIFAR-10 research training script")

    parser.add_argument("--mode", choices=["standard", "curriculum", "anticurriculum"], default="curriculum")
    parser.add_argument("--model-name", default="ResNet18")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--width-mult", type=float, default=1.0)

    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--mapping-train-csv", default="csv/mapping_train.csv")
    parser.add_argument("--mapping-test-csv", default="csv/mapping_test.csv")
    parser.add_argument("--difficulty-train-csv", default="csv/resnet18_cifar10_100pct_seed42_20260104_100055_difficulty_ordered_train.csv")
    parser.add_argument("--difficulty-test-csv", default="csv/resnet18_cifar10_100pct_seed42_20260104_095243_difficulty_ordered_test.csv")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument(
        "--scheduler-type",
        choices=["none", "cosine_annealing", "exponential", "cosine_stage", "exponential_stage"],
        default="cosine_annealing",
    )
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--stage-lr-len", type=int, default=20)
    parser.add_argument("--stage-lr-max", default="0.1,0.1,0.1,0.1,0.1,0.1")
    parser.add_argument("--stage-lr-min", default="0.0,0.0,0.0,0.0,0.0,0.0")

    parser.add_argument("--stage-epochs", type=int, default=20)
    parser.add_argument("--initial-samples", type=int, default=10000)
    parser.add_argument("--samples-per-stage", type=int, default=10000)
    parser.add_argument("--total-train-samples", type=int, default=50000)
    parser.add_argument("--test-stage-sizes", default="2000,2000,2000,2000,2000")

    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--random-horizontal-flip", action="store_true")

    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--plot-frequency", type=int, default=10)
    parser.add_argument("--run-name", default="")

    return parser


def main():
    args = build_parser().parse_args()

    if args.total_train_samples <= 0:
        raise ValueError("total_train_samples must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args.train_dir = resolve_repo_path(args.train_dir)
    args.test_dir = resolve_repo_path(args.test_dir)
    args.mapping_train_csv = resolve_repo_path(args.mapping_train_csv)
    args.mapping_test_csv = resolve_repo_path(args.mapping_test_csv)
    args.difficulty_train_csv = resolve_repo_path(args.difficulty_train_csv)
    args.difficulty_test_csv = resolve_repo_path(args.difficulty_test_csv)
    args.log_dir = resolve_repo_path(args.log_dir)
    args.plot_dir = resolve_repo_path(args.plot_dir)
    args.checkpoint_dir = resolve_repo_path(args.checkpoint_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_name.strip() or args.mode
    run_dir = os.path.join(args.log_dir, f"{run_prefix}_{timestamp}")
    plot_run_dir = os.path.join(args.plot_dir, f"{run_prefix}_{timestamp}")
    checkpoint_run_dir = os.path.join(args.checkpoint_dir, f"{run_prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_run_dir, exist_ok=True)
    os.makedirs(checkpoint_run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    log_file = os.path.join(run_dir, "training.log")
    log_fp = open(log_file, "w", encoding="utf-8")

    def log_print(*values, **kwargs):
        print(*values, **kwargs)
        print(*values, **kwargs, file=log_fp)
        log_fp.flush()

    train_transform_ops = []
    if args.random_crop:
        train_transform_ops.append(transforms.RandomCrop(32, padding=4))
    if args.random_horizontal_flip:
        train_transform_ops.append(transforms.RandomHorizontalFlip())
    train_transform_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_train = transforms.Compose(train_transform_ops)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_stage_sizes = parse_int_list(args.test_stage_sizes)

    log_print("==> Loading test data..")
    test_dataset = CIFAR10TestDataset(args.test_dir, args.mapping_test_csv, transform=transform_test)
    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    log_print("==> Creating test stages..")
    test_stages = create_test_stages(args.mapping_test_csv, args.difficulty_test_csv, test_stage_sizes)
    log_print(f"Created {len(test_stages)} test stages with sizes: {[len(stage) for stage in test_stages]}")

    log_print("==> Building model..")
    model = create_model(args.model_name, model_args={"num_classes": args.num_classes, "width_mult": args.width_mult})
    model = model.to(device)
    if device == "cuda":
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(args, optimizer)

    best_acc = 0.0
    start_epoch = 0
    checkpoint_name = f"ckpt_{args.mode}.pth" if args.mode != "standard" else "ckpt.pth"
    checkpoint_path = args.resume_path or os.path.join(checkpoint_run_dir, checkpoint_name)

    if args.resume and os.path.exists(checkpoint_path):
        log_print(f"==> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["net"])
        best_acc = float(checkpoint.get("acc", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "stage_acc": [],
        "lr": [],
        "train_time": [],
        "train_data_loading_time": [],
        "train_compute_time": [],
        "train_overhead_time": [],
        "test_time": [],
        "test_data_loading_time": [],
        "test_inference_time": [],
        "test_overhead_time": [],
        "total_time": [],
        "gpu_allocated": [],
        "gpu_reserved": [],
        "gpu_max_allocated": [],
    }

    total_training_time = 0.0
    current_stage = -1
    current_trainloader = None

    log_print("==> Starting training..")
    log_print(f"Mode: {args.mode}")
    log_print(f"Total epochs: {args.epochs}")
    log_print(f"Model: {args.model_name}")
    log_print(f"Scheduler: {args.scheduler_type}")
    log_print(f"Batch size: {args.batch_size} | Workers: {args.num_workers}")

    for epoch in range(start_epoch, args.epochs):
        if args.mode == "standard":
            if current_trainloader is None:
                train_dataset = CIFAR10RawDataset(
                    data_dir=args.train_dir,
                    mapping_csv=args.mapping_train_csv,
                    transform=transform_train,
                )
                current_trainloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=device == "cuda",
                )
        else:
            new_stage = epoch // args.stage_epochs
            if new_stage != current_stage or current_trainloader is None:
                train_indices = build_train_indices(args, epoch)
                num_samples = len(train_indices) if train_indices is not None else args.total_train_samples
                log_print(f"\n==> Epoch {epoch}: updating {args.mode} subset to {num_samples} samples (stage {new_stage + 1})")

                train_dataset = build_train_dataset(args, train_indices, transform_train)
                current_trainloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=device == "cuda",
                )
                current_stage = new_stage

        train_loss, train_acc, train_time, train_data_time, train_compute_time, train_overhead_time = train_epoch(
            model, current_trainloader, criterion, optimizer, device
        )
        test_loss, test_acc, stage_accs, test_time, test_data_time, test_inference_time, test_overhead_time = test_epoch(
            model, testloader, device, test_stages
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        gpu_info = get_gpu_memory_info(device)
        epoch_total_time = train_time + test_time
        total_training_time += epoch_total_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["stage_acc"].append(stage_accs)
        history["lr"].append(current_lr)
        history["train_time"].append(train_time)
        history["train_data_loading_time"].append(train_data_time)
        history["train_compute_time"].append(train_compute_time)
        history["train_overhead_time"].append(train_overhead_time)
        history["test_time"].append(test_time)
        history["test_data_loading_time"].append(test_data_time)
        history["test_inference_time"].append(test_inference_time)
        history["test_overhead_time"].append(test_overhead_time)
        history["total_time"].append(total_training_time)
        history["gpu_allocated"].append(gpu_info["allocated_gb"])
        history["gpu_reserved"].append(gpu_info["reserved_gb"])
        history["gpu_max_allocated"].append(gpu_info["max_allocated_gb"])

        log_print(f"\nEpoch {epoch}:")
        log_print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        log_print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        if stage_accs:
            stage_str = " | ".join(f"L{i + 1} {acc:.2f}%" for i, acc in enumerate(stage_accs))
            log_print(f"  Stage Accuracies: {stage_str}")
        log_print(f"  LR: {current_lr:.6f} | Best Acc: {best_acc:.2f}%")
        log_print(
            f"  Timing: Train={train_time:.2f}s (data={train_data_time:.2f}s, compute={train_compute_time:.2f}s, overhead={train_overhead_time:.2f}s)"
        )
        log_print(
            f"          Test={test_time:.2f}s (data={test_data_time:.2f}s, inference={test_inference_time:.2f}s, overhead={test_overhead_time:.2f}s)"
        )
        log_print(f"          Total epoch={epoch_total_time:.2f}s | Cumulative={total_training_time:.2f}s")
        if device == "cuda" and torch.cuda.is_available():
            log_print(
                f"  GPU Memory: Allocated={gpu_info['allocated_gb']:.2f}GB | Reserved={gpu_info['reserved_gb']:.2f}GB | Max Allocated={gpu_info['max_allocated_gb']:.2f}GB"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_best:
                state = {"net": model.state_dict(), "acc": test_acc, "epoch": epoch}
                torch.save(state, os.path.join(checkpoint_run_dir, checkpoint_name))
                log_print(f"  Saved checkpoint (best acc: {best_acc:.2f}%)")

        if (epoch + 1) % args.plot_frequency == 0:
            save_plots(history, plot_run_dir, epoch)
            log_print(f"  Saved plots to {plot_run_dir}")

    log_fp.close()
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
