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

from env.simulator import build_default_simulation
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


def parse_compression_schedule(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("compression schedule must contain at least one value")
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


def choose_target_link(simulator, uav_id: int) -> tuple[object, dict | None]:
    uav = next((item for item in simulator.uav_list if item.uid == uav_id), None)
    if uav is None:
        raise ValueError(f"UAV {uav_id} not found")

    results = simulator.a2g_channel.compute_sinr_and_rate(uav, simulator.ue_list, simulator.uav_list)
    if not results:
        return uav, None
    best_link = max(results, key=lambda item: item["sinr"])
    return uav, best_link


def plot_integration_metrics(rows: list[dict], ratios: list[float], output_dir: Path, prefix: str) -> Path:
    times = np.array(sorted({float(row["time"]) for row in rows}), dtype=float)
    ratio_to_rows = {ratio: [row for row in rows if abs(float(row["compression_ratio"]) - ratio) < 1e-9] for ratio in ratios}
    base_rows = ratio_to_rows[ratios[0]]

    sinr_db = np.array([float(row["sinr_db"]) if row["coverage_available"] == 1 else np.nan for row in base_rows], dtype=float)
    energy = np.array([float(row["uav_energy"]) for row in base_rows], dtype=float)
    rate_mbps = np.array([float(row["rate_bps"]) / 1e6 if row["coverage_available"] == 1 else np.nan for row in base_rows], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle("DeepJSCC In-Environment Integration Metrics", fontsize=14)

    axes[0, 0].plot(times, sinr_db, color="tab:blue", linewidth=2)
    axes[0, 0].set_title("SINR")
    axes[0, 0].set_ylabel("dB")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    for ratio in ratios:
        subset = ratio_to_rows[ratio]
        psnr = np.array([float(row["psnr"]) if row["coverage_available"] == 1 else np.nan for row in subset], dtype=float)
        axes[0, 1].plot(times, psnr, linewidth=2, label=f"ratio={ratio:.2f}")
    axes[0, 1].set_title("PSNR by Compression Ratio")
    axes[0, 1].set_ylabel("dB")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=9)

    for ratio in ratios:
        subset = ratio_to_rows[ratio]
        utility = np.array([float(row["semantic_utility"]) if row["coverage_available"] == 1 else np.nan for row in subset], dtype=float)
        axes[1, 0].plot(times, utility, linewidth=2, label=f"ratio={ratio:.2f}")
    axes[1, 0].set_title("Semantic Utility by Compression Ratio")
    axes[1, 0].set_ylabel("Utility")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend(fontsize=9)

    for ratio in ratios:
        subset = ratio_to_rows[ratio]
        cost = np.array([float(row["transmission_cost"]) if row["coverage_available"] == 1 else np.nan for row in subset], dtype=float)
        axes[1, 1].plot(times, cost, linewidth=2, label=f"cost@{ratio:.2f}")
    axes[1, 1].set_title("Transmission Cost by Compression Ratio")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend(fontsize=9)

    axes[2, 0].plot(times, energy, color="tab:brown", linewidth=2)
    axes[2, 0].set_title("UAV Energy")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("J")
    axes[2, 0].grid(True, linestyle="--", alpha=0.4)

    axes[2, 1].plot(times, rate_mbps, color="tab:cyan", linewidth=2)
    axes[2, 1].set_title("A2G Rate")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Mbps")
    axes[2, 1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, prefix, ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_integration_test(args: argparse.Namespace) -> tuple[Path, Path]:
    simulator = build_default_simulation()
    simulator.reset()

    module = DeepJSCCScenarioModule(checkpoint_path=args.checkpoint)
    reference_images, image_description = load_reference_images(args)
    args.image_description = image_description
    ratios = parse_compression_schedule(args.compression_schedule)
    output_dir = Path(args.output_dir)
    csv_path = ensure_unique_path(output_dir, prefix=args.csv_prefix, suffix=".csv")

    fieldnames = [
        "step",
        "time",
        "image_source",
        "uav_id",
        "uav_energy",
        "uav_energy_state",
        "target_ue_id",
        "sinr_linear",
        "sinr_db",
        "rate_bps",
        "compression_ratio",
        "psnr",
        "semantic_quality",
        "semantic_utility",
        "transmission_cost",
        "transmitted_symbols",
        "total_symbols",
        "coverage_available",
        "best_ratio_by_utility",
    ]
    rows: list[dict] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(args.steps):
            observation, _ = simulator.step()
            uav, best_link = choose_target_link(simulator, args.uav_id)

            per_ratio_rows = []
            best_ratio_by_utility = ""
            best_utility = -float("inf")

            if best_link is not None:
                sinr_linear = float(best_link["sinr"])
                sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-10))
                rate_bps = float(best_link["rate"])

                for compression_ratio in ratios:
                    sample_results = [
                        module.transmit(
                            image=image,
                            sinr_db=sinr_db,
                            compression_ratio=compression_ratio,
                            task_weights={"quality": args.quality_weight, "cost": args.cost_weight},
                        )
                        for image in reference_images
                    ]
                    row = {
                        "step": step,
                        "time": observation["time"],
                        "image_source": args.image_description,
                        "uav_id": uav.uid,
                        "uav_energy": float(uav.energy),
                        "uav_energy_state": uav.energy_state,
                        "target_ue_id": best_link["ue"].uid,
                        "sinr_linear": sinr_linear,
                        "sinr_db": sinr_db,
                        "rate_bps": rate_bps,
                        "compression_ratio": compression_ratio,
                        "psnr": float(np.mean([result.psnr for result in sample_results])),
                        "semantic_quality": float(np.mean([result.semantic_quality for result in sample_results])),
                        "semantic_utility": float(np.mean([result.semantic_utility for result in sample_results])),
                        "transmission_cost": float(np.mean([result.transmission_cost for result in sample_results])),
                        "transmitted_symbols": int(round(np.mean([result.transmitted_symbols for result in sample_results]))),
                        "total_symbols": int(round(np.mean([result.total_symbols for result in sample_results]))),
                        "coverage_available": 1,
                        "best_ratio_by_utility": "",
                    }
                    per_ratio_rows.append(row)
                    if row["semantic_utility"] > best_utility:
                        best_utility = row["semantic_utility"]
                        best_ratio_by_utility = f"{compression_ratio:.2f}"

                for row in per_ratio_rows:
                    row["best_ratio_by_utility"] = best_ratio_by_utility
                    writer.writerow(row)
                    rows.append(row)
                csv_file.flush()

                summary = " | ".join(
                    [f"r={row['compression_ratio']:.2f}: psnr={row['psnr']:.2f}, u={row['semantic_utility']:.3f}" for row in per_ratio_rows]
                )
                print(
                    f"step={step} time={observation['time']:.1f}s uav={uav.uid} ue={best_link['ue'].uid} "
                    f"sinr_db={sinr_db:.2f} rate={rate_bps/1e6:.2f}Mbps best_ratio={best_ratio_by_utility} state={uav.energy_state}\n"
                    f"  {summary}"
                )
            else:
                for compression_ratio in ratios:
                    row = {
                        "step": step,
                        "time": observation["time"],
                        "image_source": args.image_description,
                        "uav_id": uav.uid,
                        "uav_energy": float(uav.energy),
                        "uav_energy_state": uav.energy_state,
                        "target_ue_id": "",
                        "sinr_linear": "",
                        "sinr_db": "",
                        "rate_bps": "",
                        "compression_ratio": compression_ratio,
                        "psnr": "",
                        "semantic_quality": "",
                        "semantic_utility": "",
                        "transmission_cost": "",
                        "transmitted_symbols": "",
                        "total_symbols": "",
                        "coverage_available": 0,
                        "best_ratio_by_utility": "",
                    }
                    writer.writerow(row)
                    rows.append(row)
                csv_file.flush()
                print(f"step={step} time={observation['time']:.1f}s uav={uav.uid} no_coverage state={uav.energy_state}")

    plot_path = plot_integration_metrics(rows, ratios=ratios, output_dir=output_dir, prefix=args.plot_prefix)
    return csv_path, plot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal in-environment DeepJSCC integration test")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulator steps to run")
    parser.add_argument("--uav-id", type=int, default=1, help="UAV id to inspect")
    parser.add_argument("--compression-schedule", type=str, default="1.0,0.75,0.5,0.25", help="Comma-separated compression ratios to compare at every step")
    parser.add_argument("--quality-weight", type=float, default=1.0, help="Semantic quality weight")
    parser.add_argument("--cost-weight", type=float, default=0.05, help="Transmission cost weight")
    parser.add_argument("--checkpoint", type=str, default="semantic_jscc/checkpoints/deepjscc_best.pth", help="DeepJSCC checkpoint path")
    parser.add_argument("--output-dir", type=str, default="semantic_jscc/results", help="Directory for CSV logs")
    parser.add_argument("--csv-prefix", type=str, default="env_integration_metrics", help="CSV filename prefix")
    parser.add_argument("--plot-prefix", type=str, default="env_integration_curves", help="Plot filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the reference image")
    parser.add_argument(
        "--image-source",
        type=str,
        default="cifar",
        choices=["cifar", "random"],
        help="Reference image source for DeepJSCC calls",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="CIFAR-10 test sample index when image-source=cifar")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of reference samples to average per step")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path, plot_path = run_integration_test(args)
    print(f"reference_image={args.image_description}")
    print(f"saved_csv(env_integration_metrics)={csv_path}")
    print(f"saved_plot(env_integration_curves)={plot_path}")


if __name__ == "__main__":
    main()
