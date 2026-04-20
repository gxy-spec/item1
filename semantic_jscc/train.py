from __future__ import annotations
import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_jscc.models import DeepJSCCModel


def parse_ratio_list(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("keep-ratios must contain at least one value")
    return values


def ensure_unique_path(output_dir: str | Path, prefix: str, suffix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{prefix}_{timestamp}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{prefix}_{timestamp}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def get_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, keep_ratios):
    model.train()
    total_loss = 0.0
    ratio_stats = {ratio: [] for ratio in keep_ratios}

    for images, _ in train_loader:
        images = images.to(device)
        keep_ratio = random.choice(keep_ratios)
        optimizer.zero_grad()
        recon = model(images, compression_ratio=keep_ratio)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        ratio_stats[keep_ratio].append(loss.item())

    avg_ratio_stats = {
        ratio: (sum(losses) / len(losses) if losses else float("nan"))
        for ratio, losses in ratio_stats.items()
    }
    return total_loss / len(train_loader.dataset), avg_ratio_stats


def evaluate(model, test_loader, criterion, device, keep_ratios):
    model.eval()
    total_loss = 0.0
    ratio_losses = {ratio: 0.0 for ratio in keep_ratios}
    ratio_counts = {ratio: 0 for ratio in keep_ratios}

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_losses = []
            for keep_ratio in keep_ratios:
                recon = model(images, compression_ratio=keep_ratio)
                loss = criterion(recon, images)
                loss_value = loss.item()
                batch_losses.append(loss_value)
                ratio_losses[keep_ratio] += loss_value * images.size(0)
                ratio_counts[keep_ratio] += images.size(0)
            total_loss += (sum(batch_losses) / len(batch_losses)) * images.size(0)

    avg_ratio_losses = {
        ratio: (ratio_losses[ratio] / max(ratio_counts[ratio], 1))
        for ratio in keep_ratios
    }
    return total_loss / len(test_loader.dataset), avg_ratio_losses


def format_ratio_metrics(prefix: str, metrics: dict[float, float]) -> str:
    ordered = sorted(metrics.items(), key=lambda item: item[0], reverse=True)
    return " ".join([f"{prefix}@{ratio:.2f}={value:.6f}" for ratio, value in ordered])


def plot_training_curves(rows: list[dict[str, float]], keep_ratios: list[float], output_dir: str | Path) -> Path:
    plot_path = ensure_unique_path(output_dir, "deepjscc_training", ".png")
    epochs = [int(row["epoch"]) for row in rows]
    train_mse = [float(row["train_loss"]) for row in rows]
    val_mse = [float(row["val_loss"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("DeepJSCC Training Curves", fontsize=14)

    axes[0, 0].plot(epochs, train_mse, label="train_mse", linewidth=2)
    axes[0, 0].plot(epochs, val_mse, label="val_mse_avg", linewidth=2)
    axes[0, 0].set_title("Overall MSE")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend()

    for ratio in sorted(keep_ratios, reverse=True):
        train_values = [float(row[f"train_ratio_{ratio:.2f}"]) for row in rows]
        axes[0, 1].plot(epochs, train_values, linewidth=2, label=f"train@{ratio:.2f}")
    axes[0, 1].set_title("Train MSE by Keep Ratio")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=9)

    for ratio in sorted(keep_ratios, reverse=True):
        val_values = [float(row[f"val_ratio_{ratio:.2f}"]) for row in rows]
        axes[1, 0].plot(epochs, val_values, linewidth=2, label=f"val@{ratio:.2f}")
    axes[1, 0].set_title("Validation MSE by Keep Ratio")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend(fontsize=9)

    best_val_so_far = []
    current_best = float("inf")
    for value in val_mse:
        current_best = min(current_best, value)
        best_val_so_far.append(current_best)
    axes[1, 1].plot(epochs, best_val_so_far, label="best_val_mse_so_far", linewidth=2, color="tab:red")
    axes[1, 1].set_title("Best Validation MSE So Far")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="DeepJSCC CIFAR-10 Training with keep-ratio awareness")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--snr-db",
        "--sinr-db",
        dest="sinr_db",
        type=float,
        default=10.0,
        help="SINR value in dB for the channel model",
    )
    parser.add_argument(
        "--keep-ratios",
        type=str,
        default="1.0,0.75,0.5,0.25",
        help="Comma-separated keep ratios used during training and validation",
    )
    parser.add_argument("--output-dir", type=str, default="semantic_jscc/checkpoints")
    parser.add_argument("--log-dir", type=str, default="semantic_jscc/results")
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="uniform",
        choices=["uniform", "prefix"],
        help="Channel selection strategy used for keep-ratio masking",
    )
    args = parser.parse_args()

    keep_ratios = parse_ratio_list(args.keep_ratios)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = DeepJSCCModel(latent_dim=args.latent_dim, sinr_db=args.sinr_db, mask_mode=args.mask_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Device:", device)
    print("Training DeepJSCC model with SINR =", args.sinr_db, "dB")
    print("Keep ratios:", keep_ratios)
    print("Mask mode:", args.mask_mode)

    csv_path = ensure_unique_path(args.log_dir, "deepjscc_training", ".csv")
    fieldnames = ["epoch", "train_loss", "val_loss"]
    for ratio in sorted(keep_ratios, reverse=True):
        fieldnames.append(f"train_ratio_{ratio:.2f}")
    for ratio in sorted(keep_ratios, reverse=True):
        fieldnames.append(f"val_ratio_{ratio:.2f}")

    rows: list[dict[str, float]] = []
    best_loss = float("inf")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_ratio_stats = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                keep_ratios=keep_ratios,
            )
            val_loss, val_ratio_losses = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                keep_ratios=keep_ratios,
            )
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE(avg): {val_loss:.6f} | "
                f"{format_ratio_metrics('train', train_ratio_stats)} | "
                f"{format_ratio_metrics('val', val_ratio_losses)}"
            )

            row: dict[str, float] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            for ratio in sorted(keep_ratios, reverse=True):
                row[f"train_ratio_{ratio:.2f}"] = float(train_ratio_stats[ratio])
            for ratio in sorted(keep_ratios, reverse=True):
                row[f"val_ratio_{ratio:.2f}"] = float(val_ratio_losses[ratio])
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, "deepjscc_best.pth")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "sinr_db": args.sinr_db,
                        "keep_ratios": keep_ratios,
                        "latent_dim": args.latent_dim,
                        "mask_mode": args.mask_mode,
                    },
                    checkpoint_path,
                )
                print("Saved best checkpoint:", checkpoint_path)

    plot_path = plot_training_curves(rows, keep_ratios, args.log_dir)
    print(f"saved_csv(deepjscc_training)={csv_path}")
    print(f"saved_plot(deepjscc_training)={plot_path}")


if __name__ == "__main__":
    main()
