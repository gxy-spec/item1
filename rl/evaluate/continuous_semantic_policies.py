from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch

from rl.agents import SACAgent
from rl.baselines.continuous_semantic_env import run_semantic_policy_episode
from rl.envs import ContinuousSemanticSAoIEnv, ContinuousSemanticSAoIEnvConfig


def add_experiment_caption(fig: plt.Figure, experiment_name: str) -> None:
    fig.text(0.5, 0.012, f"Experiment: {experiment_name}", ha="center", va="bottom", fontsize=10)


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
        "avg_mean_saoi",
        "avg_mean_aoi",
        "final_energy",
        "charge_steps",
        "success_updates",
        "service_attempts",
        "success_rate",
        "avg_queue",
        "final_queue",
        "max_queue",
    ]
    return {key: mean(float(item[key]) for item in results) for key in keys}


def run_semantic_sac_episode(model_path: str, config: ContinuousSemanticSAoIEnvConfig, seed: int) -> dict[str, float]:
    env = ContinuousSemanticSAoIEnv(config)
    obs = env.reset(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
    )
    agent.load(model_path)

    total_reward = 0.0
    mean_saois = []
    mean_aois = []
    charge_steps = 0
    success_updates = 0
    service_attempts = 0
    queue_values = []
    max_queue = 0.0
    done = False

    while not done:
        action = agent.select_action(obs, evaluate=True).astype("float32")
        obs, reward, done, info = env.step(action)
        total_reward += reward
        mean_saois.append(info["mean_saoi"])
        mean_aois.append(info["mean_aoi"])
        if info["energy_state"] in {"return", "charging", "resume"}:
            charge_steps += 1
        success_updates += int(info.get("success_count", 1 if info["success"] else 0))
        service_attempts += int(info.get("selected_user_count", 1 if info["selected_ue"] is not None else 0))
        queue_values.append(float(info["virtual_energy_queue"]))
        max_queue = max(max_queue, float(info["virtual_energy_queue"]))

    return {
        "policy": "semantic_sac",
        "episode_reward": total_reward,
        "avg_mean_saoi": float(sum(mean_saois) / len(mean_saois)) if mean_saois else 0.0,
        "avg_mean_aoi": float(sum(mean_aois) / len(mean_aois)) if mean_aois else 0.0,
        "final_energy": float(env.uav.energy),
        "final_state": env.uav.energy_state,
        "charge_steps": charge_steps,
        "success_updates": success_updates,
        "service_attempts": service_attempts,
        "success_rate": success_updates / max(service_attempts, 1),
        "avg_queue": float(sum(queue_values) / len(queue_values)) if queue_values else 0.0,
        "final_queue": float(queue_values[-1]) if queue_values else 0.0,
        "max_queue": max_queue,
    }


def plot_policy_comparison(
    all_rows: list[dict[str, float]],
    output_dir: Path,
    experiment_name: str = "continuous_semantic_policy_evaluation",
) -> Path:
    policies = sorted({str(row["policy"]) for row in all_rows})
    metrics = [
        ("episode_reward", "Reward", "Reward (a.u.)", False),
        ("avg_mean_saoi", "Average SAoI", "SAoI (time slots)", True),
        ("avg_mean_aoi", "Average AoI", "AoI (time slots)", True),
        ("success_rate", "Success Rate", "Success rate (-)", False),
        ("final_energy", "Final Energy", "Energy (J)", False),
        ("avg_queue", "Average Queue", "Queue (J)", True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Continuous Semantic SAoI Policy Comparison", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (metric_key, title, ylabel, lower_is_better) in zip(flat_axes, metrics):
        metric_values = []
        for policy in policies:
            values = [float(row[metric_key]) for row in all_rows if str(row["policy"]) == policy]
            metric_values.append(mean(values) if values else 0.0)
        bars = ax.bar(policies, metric_values, color=["tab:blue", "tab:green", "tab:orange"][: len(policies)])
        arrow = "↓" if lower_is_better else "↑"
        ax.set_title(f"{title} ({arrow})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_xlabel("Baseline (-)")
        ax.set_ylabel(ylabel)
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    add_experiment_caption(fig, experiment_name)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, "continuous_semantic_policy_evaluation", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic continuous baselines")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/evaluation/continuous_semantic")
    parser.add_argument("--semantic-sac-model", type=str, default=None)
    parser.add_argument("--multi-user-association", action="store_true")
    parser.add_argument("--association-threshold", type=float, default=0.5)
    parser.add_argument("--resource-allocation-mode", type=str, default="uniform", choices=["fixed", "uniform", "kkt"])
    args = parser.parse_args()

    config = ContinuousSemanticSAoIEnvConfig(
        max_steps=args.max_steps,
        num_ues=args.num_ues,
        seed=args.seed,
        multi_user_association=args.multi_user_association,
        association_threshold=args.association_threshold,
        resource_allocation_mode=args.resource_allocation_mode,
    )
    policies = ["random_semantic_continuous", "continuous_rule_semantic"]
    if args.semantic_sac_model:
        policies.append("semantic_sac")
    all_rows = []

    for policy in policies:
        episode_results = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            if policy == "semantic_sac":
                result = run_semantic_sac_episode(model_path=args.semantic_sac_model, config=config, seed=seed)
            else:
                result = run_semantic_policy_episode(policy_name=policy, config=config, seed=seed)
            result["episode"] = episode + 1
            episode_results.append(result)
            all_rows.append(result)

        summary = summarise(episode_results)
        print(
            f"policy={policy} reward={summary['episode_reward']:.3f} avg_saoi={summary['avg_mean_saoi']:.3f} "
            f"avg_aoi={summary['avg_mean_aoi']:.3f} success_rate={summary['success_rate']:.3f} "
            f"final_energy={summary['final_energy']:.1f} avg_queue={summary['avg_queue']:.1f}"
        )

    output_dir = Path(args.output_dir)
    csv_path = ensure_unique_path(output_dir, "continuous_semantic_policy_evaluation", ".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "policy",
                "episode",
                "episode_reward",
                "avg_mean_saoi",
                "avg_mean_aoi",
                "final_energy",
                "final_state",
                "charge_steps",
                "success_updates",
                "service_attempts",
                "success_rate",
                "avg_queue",
                "final_queue",
                "max_queue",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved_csv(continuous_semantic_policy_evaluation)={csv_path}")

    experiment_name = f"semantic_eval_{args.resource_allocation_mode}_{'multi' if args.multi_user_association else 'single'}"
    plot_path = plot_policy_comparison(all_rows, output_dir=output_dir, experiment_name=experiment_name)
    print(f"saved_plot(continuous_semantic_policy_evaluation)={plot_path}")


if __name__ == "__main__":
    main()
