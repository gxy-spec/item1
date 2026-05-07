from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


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


def add_experiment_caption(fig: plt.Figure, experiment_name: str) -> None:
    fig.text(0.5, 0.012, f"Experiment: {experiment_name}", ha="center", va="bottom", fontsize=10)


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarise_semantic_sac(path: str | Path) -> dict[str, float]:
    rows = [row for row in load_rows(path) if row["policy"] == "semantic_sac"]
    return {
        "reward": mean(float(row["episode_reward"]) for row in rows),
        "avg_saoi": mean(float(row["avg_mean_saoi"]) for row in rows),
        "avg_queue": mean(float(row["avg_queue"]) for row in rows),
        "final_energy": mean(float(row["final_energy"]) for row in rows),
    }


def plot_comparison(
    fixed_csv: str | Path,
    uniform_csv: str | Path,
    kkt_csv: str | Path,
    output_dir: str | Path,
    experiment_name: str,
) -> Path:
    summaries = {
        "Fixed": summarise_semantic_sac(fixed_csv),
        "Uniform": summarise_semantic_sac(uniform_csv),
        "KKT": summarise_semantic_sac(kkt_csv),
    }
    labels = list(summaries.keys())
    metrics = [
        ("avg_saoi", "Average SAoI", "SAoI (time slots)", True),
        ("reward", "Reward", "Reward (a.u.)", False),
        ("avg_queue", "Average Queue", "Queue (J)", True),
        ("final_energy", "Final Energy", "Energy (J)", False),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Resource Allocation Baseline Comparison", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (key, title, ylabel, lower_is_better) in zip(flat_axes, metrics):
        values = [summaries[label][key] for label in labels]
        bars = ax.bar(labels, values, color=colors)
        arrow = "↓" if lower_is_better else "↑"
        ax.set_title(f"{title} ({arrow})")
        ax.set_xlabel("Resource allocation mode (-)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_experiment_caption(fig, experiment_name)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    output_path = ensure_unique_path(Path(output_dir), "resource_allocation_comparison", ".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot fixed/uniform/KKT comparison for semantic SAC")
    parser.add_argument("--fixed-csv", type=str, required=True)
    parser.add_argument("--uniform-csv", type=str, required=True)
    parser.add_argument("--kkt-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/comparison")
    parser.add_argument("--experiment-name", type=str, default="Semantic SAC resource allocation comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_path = plot_comparison(
        fixed_csv=args.fixed_csv,
        uniform_csv=args.uniform_csv,
        kkt_csv=args.kkt_csv,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    print(f"saved_plot(resource_allocation_comparison)={plot_path}")


if __name__ == "__main__":
    main()
