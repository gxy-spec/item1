from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training curves from a CSV log file")
    parser.add_argument("--csv", type=str, default="rl/outputs/training/logs/training_metrics.csv", help="Path to the training metrics CSV file")
    parser.add_argument("--output-dir", type=str, default="rl/outputs/training/plots", help="Directory to save generated plots")
    parser.add_argument("--prefix", type=str, default=None, help="Optional filename prefix. Defaults to the CSV stem.")
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
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        data: Dict[str, List[float]] = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                value = row[name]
                try:
                    data[name].append(float(value))
                except (TypeError, ValueError):
                    # Skip non-numeric metadata columns such as saved model paths.
                    continue
    return data


def plot_training_curves(csv_path: str | Path, output_dir: str | Path = "rl/outputs/training/plots", prefix: str | None = None) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = load_csv(csv_path)
    if not data.get("episode"):
        raise ValueError(f"CSV file is empty: {csv_path}")

    episodes = data["episode"]
    prefix = prefix or csv_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    base_name = f"{prefix}_{timestamp}"

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
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
    axes[1, 0].set_ylabel("Queue")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend()

    axes[1, 1].plot(episodes, data["success_rate"], label="success_rate", color="tab:green", linewidth=2)
    axes[1, 1].set_title("Update Success Rate")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend()

    axes[2, 0].plot(episodes, data["final_energy"], label="final_energy", color="tab:orange", linewidth=2)
    axes[2, 0].plot(episodes, data["min_energy"], label="min_energy", color="tab:brown", alpha=0.75)
    axes[2, 0].set_title("Energy")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Energy")
    axes[2, 0].grid(True, linestyle="--", alpha=0.4)
    axes[2, 0].legend()

    axes[2, 1].plot(episodes, data["charge_steps"], label="charge_steps", color="tab:purple", linewidth=2)
    axes[2, 1].set_title("Charging-related Steps")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Steps")
    axes[2, 1].grid(True, linestyle="--", alpha=0.4)
    axes[2, 1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    summary_path = ensure_unique_path(output_dir, f"{base_name}_summary")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary_path


def main() -> None:
    args = build_parser().parse_args()
    summary_path = plot_training_curves(csv_path=args.csv, output_dir=args.output_dir, prefix=args.prefix)
    print(f"saved_plot(training_curves)={summary_path}")


if __name__ == "__main__":
    main()
