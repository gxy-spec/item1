from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch

from rl.agents import SACAgent
from rl.baselines.continuous_assoc_env import run_assoc_policy_episode
from rl.envs import ContinuousAssocAoIEnvConfig, ContinuousAssocSingleUAVAoIEnv


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
        "avg_queue",
        "final_queue",
        "max_queue",
    ]
    return {key: mean(float(item[key]) for item in results) for key in keys}


def run_assoc_sac_episode(model_path: str, config: ContinuousAssocAoIEnvConfig, seed: int) -> dict[str, float]:
    env = ContinuousAssocSingleUAVAoIEnv(config)
    obs = env.reset(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
    )
    try:
        agent.load(model_path)
    except RuntimeError as exc:
        raise ValueError(
            "assoc_sac 模型加载失败。通常是因为你用了错误的模型文件，"
            "或者评估时的 --num-ues 与训练该模型时不一致。"
            "请使用 assoc_sac_best_*.pth，并确保训练和评估的 num_ues 相同。"
        ) from exc

    total_reward = 0.0
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
        mean_aois.append(info["mean_aoi"])
        if info["energy_state"] in {"return", "charging", "resume"}:
            charge_steps += 1
        if info["success"]:
            success_updates += 1
        if info["selected_ue"] is not None:
            service_attempts += 1
        queue_values.append(float(info["virtual_energy_queue"]))
        max_queue = max(max_queue, float(info["virtual_energy_queue"]))

    return {
        "policy": "assoc_sac",
        "episode_reward": total_reward,
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


def plot_policy_comparison(all_rows: list[dict[str, float]], output_dir: Path) -> Path:
    policies = sorted({str(row["policy"]) for row in all_rows})
    metrics = [
        ("episode_reward", "Reward", False),
        ("avg_mean_aoi", "Average AoI", True),
        ("success_rate", "Success Rate", False),
        ("charge_steps", "Charging-related Steps", True),
        ("final_energy", "Final Energy", False),
        ("avg_queue", "Average Queue", True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Continuous Assoc Policy Comparison", fontsize=14)
    flat_axes = axes.flatten()

    for ax, (metric_key, title, lower_is_better) in zip(flat_axes, metrics):
        metric_values = []
        for policy in policies:
            values = [float(row[metric_key]) for row in all_rows if str(row["policy"]) == policy]
            metric_values.append(mean(values) if values else 0.0)
        bars = ax.bar(policies, metric_values, color=["tab:blue", "tab:green", "tab:orange"][: len(policies)])
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Lower is better" if lower_is_better else "Higher is better")
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = ensure_unique_path(output_dir, "continuous_assoc_policy_evaluation", ".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate user-association continuous baselines")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/evaluation/continuous_assoc")
    parser.add_argument("--assoc-sac-model", type=str, default=None)
    args = parser.parse_args()

    config = ContinuousAssocAoIEnvConfig(max_steps=args.max_steps, num_ues=args.num_ues, seed=args.seed)
    policies = ["random_assoc_continuous", "continuous_rule_assoc_continuous"]
    if args.assoc_sac_model:
        policies.append("assoc_sac")
    all_rows = []

    for policy in policies:
        episode_results = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            if policy == "assoc_sac":
                result = run_assoc_sac_episode(model_path=args.assoc_sac_model, config=config, seed=seed)
            else:
                result = run_assoc_policy_episode(policy_name=policy, config=config, seed=seed)
            result["episode"] = episode + 1
            episode_results.append(result)
            all_rows.append(result)

        summary = summarise(episode_results)
        print(
            f"policy={policy} reward={summary['episode_reward']:.3f} avg_aoi={summary['avg_mean_aoi']:.3f} "
            f"success_rate={summary['success_rate']:.3f} charge_steps={summary['charge_steps']:.1f} "
            f"final_energy={summary['final_energy']:.1f} avg_queue={summary['avg_queue']:.1f}"
        )

    output_dir = Path(args.output_dir)
    csv_path = ensure_unique_path(output_dir, "continuous_assoc_policy_evaluation", ".csv")
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
                "avg_queue",
                "final_queue",
                "max_queue",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved_csv(continuous_assoc_policy_evaluation)={csv_path}")

    plot_path = plot_policy_comparison(all_rows, output_dir=output_dir)
    print(f"saved_plot(continuous_assoc_policy_evaluation)={plot_path}")


if __name__ == "__main__":
    main()
