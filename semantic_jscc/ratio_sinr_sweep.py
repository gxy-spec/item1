from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

matplotlib.use("Agg")

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_jscc import DeepJSCCScenarioModule


def ensure_unique_path(output_dir: Path, prefix: str, suffix: str) -> Path:
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


def parse_float_list(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("input list must contain at least one numeric value")
    return values


def build_random_reference_image(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((3, 32, 32), generator=generator)


def load_cifar_test_image(sample_index: int) -> tuple[torch.Tensor, int]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    image, label = dataset[sample_index % len(dataset)]
    return image, int(label)


def load_reference_images(args: argparse.Namespace) -> tuple[list[torch.Tensor], str]:
    if args.image_source == "random":
        images = [build_random_reference_image(args.seed + offset) for offset in range(args.num_samples)]
        return images, f"random(seed={args.seed}, count={args.num_samples})"

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    images = []
    labels = []
    for offset in range(args.num_samples):
        image, label = dataset[(args.sample_index + offset) % len(dataset)]
        images.append(image)
        labels.append(int(label))
    return images, f"cifar10_test(start={args.sample_index}, count={args.num_samples}, labels={labels})"


def plot_sweep(rows: list[dict], ratios: list[float], output_dir: Path, prefix: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("DeepJSCC Keep-Ratio vs SINR Sweep", fontsize=14)

    for ratio in ratios:
        subset = [row for row in rows if abs(float(row["compression_ratio"]) - ratio) < 1e-9]
        sinr_db = np.array([float(row["sinr_db"]) for row in subset], dtype=float)
        psnr = np.array([float(row["psnr"]) for row in subset], dtype=float)
        utility = np.array([float(row["semantic_utility"]) for row in subset], dtype=float)
        quality = np.array([float(row["semantic_quality"]) for row in subset], dtype=float)
        cost = np.array([float(row["transmission_cost"]) for row in subset], dtype=float)

        axes[0, 0].plot(sinr_db, psnr, marker="o", linewidth=2, label=f"ratio={ratio:.2f}")
        axes[0, 1].plot(sinr_db, utility, marker="o", linewidth=2, label=f"ratio={ratio:.2f}")
        axes[1, 0].plot(sinr_db, quality, marker="o", linewidth=2, label=f"ratio={ratio:.2f}")
        axes[1, 1].plot(sinr_db, cost, marker="o", linewidth=2, label=f"ratio={ratio:.2f}")

    axes[0, 0].set_title("PSNR by Keep Ratio")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("Semantic Utility by Keep Ratio")
    axes[0, 1].set_ylabel("Utility")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].set_title("Semantic Quality by Keep Ratio")
    axes[1, 0].set_xlabel("SINR (dB)")
    axes[1, 0].set_ylabel("Quality")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].set_title("Transmission Cost by Keep Ratio")
    axes[1, 1].set_xlabel("SINR (dB)")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend(fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, prefix, ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_sweep(args: argparse.Namespace) -> tuple[Path, Path]:
    sinr_values = parse_float_list(args.sinr_db_list)
    ratios = parse_float_list(args.compression_ratios)
    output_dir = Path(args.output_dir)
    csv_path = ensure_unique_path(output_dir, args.csv_prefix, ".csv")

    module = DeepJSCCScenarioModule(checkpoint_path=args.checkpoint)
    images, image_description = load_reference_images(args)
    args.image_description = image_description

    fieldnames = [
        "image_source",
        "sinr_db",
        "compression_ratio",
        "psnr",
        "mse",
        "semantic_quality",
        "semantic_utility",
        "transmission_cost",
        "transmitted_symbols",
        "total_symbols",
    ]
    rows: list[dict] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sinr_db in sinr_values:
            step_rows = []
            for ratio in ratios:
                sample_results = [
                    module.transmit(
                        image=image,
                        sinr_db=sinr_db,
                        compression_ratio=ratio,
                        task_weights={"quality": args.quality_weight, "cost": args.cost_weight},
                    )
                    for image in images
                ]
                row = {
                    "sinr_db": sinr_db,
                    "image_source": args.image_description,
                    "compression_ratio": ratio,
                    "psnr": float(np.mean([result.psnr for result in sample_results])),
                    "mse": float(np.mean([result.mse for result in sample_results])),
                    "semantic_quality": float(np.mean([result.semantic_quality for result in sample_results])),
                    "semantic_utility": float(np.mean([result.semantic_utility for result in sample_results])),
                    "transmission_cost": float(np.mean([result.transmission_cost for result in sample_results])),
                    "transmitted_symbols": int(round(np.mean([result.transmitted_symbols for result in sample_results]))),
                    "total_symbols": int(round(np.mean([result.total_symbols for result in sample_results]))),
                }
                step_rows.append(row)
                writer.writerow(row)
                rows.append(row)
            csv_file.flush()

            summary = " | ".join(
                [
                    f"r={row['compression_ratio']:.2f}: psnr={row['psnr']:.2f}, utility={row['semantic_utility']:.3f}"
                    for row in step_rows
                ]
            )
            print(f"sinr_db={sinr_db:.2f} -> {summary}")

    plot_path = plot_sweep(rows, ratios=ratios, output_dir=output_dir, prefix=args.plot_prefix)
    return csv_path, plot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep DeepJSCC keep ratios under fixed SINR values")
    parser.add_argument("--sinr-db-list", type=str, default="-10,-5,0,5,10", help="Comma-separated SINR values in dB")
    parser.add_argument("--compression-ratios", type=str, default="1.0,0.75,0.5,0.25", help="Comma-separated keep ratios")
    parser.add_argument("--quality-weight", type=float, default=1.0, help="Semantic quality weight")
    parser.add_argument("--cost-weight", type=float, default=0.05, help="Transmission cost weight")
    parser.add_argument("--checkpoint", type=str, default="semantic_jscc/checkpoints/deepjscc_best.pth", help="DeepJSCC checkpoint path")
    parser.add_argument("--output-dir", type=str, default="semantic_jscc/results_test", help="Directory for CSV logs and plots")
    parser.add_argument("--csv-prefix", type=str, default="ratio_sinr_sweep_metrics", help="CSV filename prefix")
    parser.add_argument("--plot-prefix", type=str, default="ratio_sinr_sweep_curves", help="Plot filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the reference image")
    parser.add_argument(
        "--image-source",
        type=str,
        default="cifar",
        choices=["cifar", "random"],
        help="Reference image source for the sweep",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="CIFAR-10 test sample index when image-source=cifar")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of reference samples to average")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path, plot_path = run_sweep(args)
    print(f"reference_image={args.image_description}")
    print(f"saved_csv(ratio_sinr_sweep)={csv_path}")
    print(f"saved_plot(ratio_sinr_sweep)={plot_path}")


if __name__ == "__main__":
    main()
