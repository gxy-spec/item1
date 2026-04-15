from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training curves from a CSV log file")
    parser.add_argument(
        "--csv",
        type=str,
        default="rl/results/training_metrics.csv",
        help="Path to the training metrics CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rl/plots",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to the CSV stem.",
    )
    return parser


def ensure_unique_path(output_dir: Path, base_name: str, suffix: str = ".png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{base_name}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def load_csv(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        data: Dict[str, List[float]] = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                value = row[name]
                data[name].append(float(value))
    return data


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = load_csv(csv_path)
    if not data.get("episode"):
        raise ValueError(f"CSV file is empty: {csv_path}")

    episodes = data["episode"]
    prefix = args.prefix or csv_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"
    output_dir = Path(args.output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Training Curves", fontsize=14)

    axes[0, 0].plot(episodes, data["reward"], label="reward", alpha=0.45)
    axes[0, 0].plot(episodes, data["reward_ma20"], label="reward_ma20", linewidth=2)
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, data["avg_aoi"], label="avg_aoi", alpha=0.45)
    axes[0, 1].plot(episodes, data["avg_aoi_ma20"], label="avg_aoi_ma20", linewidth=2)
    axes[0, 1].set_title("Average AoI")
    axes[0, 1].set_ylabel("AoI")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend()

    axes[1, 0].plot(episodes, data["queue"], label="queue", color="tab:red", linewidth=2)
    axes[1, 0].set_title("Virtual Energy Queue")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Queue")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, data["success_rate"], label="success_rate", color="tab:green", linewidth=2)
    axes[1, 1].plot(episodes, data["final_energy"], label="final_energy", color="tab:orange", alpha=0.8)
    axes[1, 1].set_title("Success Rate / Final Energy")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    summary_path = ensure_unique_path(output_dir, f"{base_name}_summary")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"saved_plot(训练曲线图)={summary_path}")


if __name__ == "__main__":
    main()
