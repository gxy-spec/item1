from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from rl.baselines.continuous_env import run_continuous_policy_episode
from rl.envs import ContinuousAoIEnvConfig


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


def summarise(results: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "episode_reward",
        "avg_mean_aoi",
        "final_energy",
        "charge_steps",
        "success_updates",
        "service_attempts",
        "success_rate",
        "max_queue",
    ]
    summary = {}
    for key in keys:
        summary[key] = mean(float(item[key]) for item in results)
    return summary


def plot_policy_comparison(all_rows: list[dict[str, float]], output_dir: Path) -> Path:
    policies = sorted({str(row["policy"]) for row in all_rows})
    metrics = [
        ("episode_reward", "Reward", False),
        ("avg_mean_aoi", "Average AoI", True),
        ("success_rate", "Success Rate", False),
        ("charge_steps", "Charging-related Steps", True),
        ("final_energy", "Final Energy", False),
        ("max_queue", "Max Queue", True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Continuous Environment Policy Comparison", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (metric_key, title, lower_is_better) in zip(flat_axes, metrics):
        metric_values = []
        for policy in policies:
            values = [float(row[metric_key]) for row in all_rows if str(row["policy"]) == policy]
            metric_values.append(mean(values) if values else 0.0)

        bars = ax.bar(
            policies,
            metric_values,
            color=["tab:blue", "tab:green", "tab:orange", "tab:red"][: len(policies)],
        )
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Lower is better" if lower_is_better else "Higher is better")

        for bar, value in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, "continuous_policy_evaluation", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate continuous-action baselines on the AoI-energy environment")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/evaluation/continuous")
    args = parser.parse_args()

    config = ContinuousAoIEnvConfig(max_steps=args.max_steps, num_ues=args.num_ues, seed=args.seed)
    policies = ["random_continuous", "continuous_rule_continuous"]
    all_rows = []

    for policy in policies:
        episode_results = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            result = run_continuous_policy_episode(policy_name=policy, config=config, seed=seed)
            result["episode"] = episode + 1
            episode_results.append(result)
            all_rows.append(result)

        summary = summarise(episode_results)
        print(
            f"policy={policy} "
            f"reward={summary['episode_reward']:.3f} "
            f"avg_aoi={summary['avg_mean_aoi']:.3f} "
            f"success_rate={summary['success_rate']:.3f} "
            f"charge_steps={summary['charge_steps']:.1f} "
            f"final_energy={summary['final_energy']:.1f} "
            f"max_queue={summary['max_queue']:.1f}"
        )

    output_dir = Path(args.output_dir)
    csv_path = ensure_unique_path(output_dir, "continuous_policy_evaluation", ".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "policy",
                "episode",
                "episode_reward",
                "avg_mean_aoi",
                "final_energy",
                "final_state",
                "charge_steps",
                "success_updates",
                "service_attempts",
                "success_rate",
                "max_queue",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved_csv(continuous_policy_evaluation)={csv_path}")

    plot_path = plot_policy_comparison(all_rows, output_dir=output_dir)
    print(f"saved_plot(continuous_policy_evaluation)={plot_path}")


if __name__ == "__main__":
    main()
