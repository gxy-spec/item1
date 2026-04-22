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


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarise_eval_policy(path: str | Path, policy_name: str, reward_key: str, aoi_key: str, queue_key: str) -> dict[str, float]:
    rows = [r for r in load_csv_rows(path) if r["policy"] == policy_name]
    return {
        "policy": policy_name,
        "reward": mean(float(r[reward_key]) for r in rows),
        "avg_aoi": mean(float(r[aoi_key]) for r in rows),
        "success_rate": mean(float(r["success_rate"]) for r in rows),
        "charge_steps": mean(float(r["charge_steps"]) for r in rows),
        "final_energy": mean(float(r["final_energy"]) for r in rows),
        "queue": mean(float(r[queue_key]) for r in rows),
    }


def plot_comparison(summaries: list[dict[str, float]], output_dir: Path) -> Path:
    metrics = [
        ("reward", "Reward", False),
        ("avg_aoi", "Average AoI", True),
        ("success_rate", "Success Rate", False),
        ("charge_steps", "Charging-related Steps", True),
        ("final_energy", "Final Energy", False),
        ("queue", "Queue", True),
    ]
    labels = [item["policy"] for item in summaries]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Baseline Comparison (DQN vs SAC vs Assoc-SAC)", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (key, title, lower_is_better) in zip(flat_axes, metrics):
        values = [item[key] for item in summaries]
        bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"][: len(values)])
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Lower is better" if lower_is_better else "Higher is better")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.text(
        0.02,
        0.01,
        "Note: All methods use evaluation summaries. DQN uses the discrete evaluation CSV, SAC/Assoc-SAC use continuous evaluation CSVs.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, "baseline_comparison", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN, SAC and Assoc-SAC baselines")
    parser.add_argument("--dqn-eval-csv", type=str, required=True)
    parser.add_argument("--sac-eval-csv", type=str, required=True)
    parser.add_argument("--assoc-sac-eval-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/comparison")
    args = parser.parse_args()

    summaries = [
        summarise_eval_policy(args.dqn_eval_csv, "dqn", "episode_reward", "avg_mean_aoi", "avg_queue"),
        summarise_eval_policy(args.sac_eval_csv, "sac", "episode_reward", "avg_mean_aoi", "avg_queue"),
        summarise_eval_policy(args.assoc_sac_eval_csv, "assoc_sac", "episode_reward", "avg_mean_aoi", "avg_queue"),
    ]

    for item in summaries:
        print(
            f"policy={item['policy']} reward={item['reward']:.3f} avg_aoi={item['avg_aoi']:.3f} "
            f"success_rate={item['success_rate']:.3f} charge_steps={item['charge_steps']:.3f} "
            f"final_energy={item['final_energy']:.3f} queue={item['queue']:.3f}"
        )

    plot_path = plot_comparison(summaries, Path(args.output_dir))
    print(f"saved_plot(baseline_comparison)={plot_path}")


if __name__ == "__main__":
    main()
