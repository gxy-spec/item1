from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.agents import SACAgent
from rl.baselines.continuous_semantic_env import continuous_rule_semantic_policy, random_semantic_continuous_policy
from rl.envs import ContinuousSemanticSAoIEnv, ContinuousSemanticSAoIEnvConfig


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


def build_semantic_sac_agent(env: ContinuousSemanticSAoIEnv, model_path: str) -> SACAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
    )
    agent.load(model_path)
    return agent


def select_action(env: ContinuousSemanticSAoIEnv, obs: np.ndarray, policy: str, agent: SACAgent | None = None) -> np.ndarray:
    if policy == "rule":
        return continuous_rule_semantic_policy(env, obs)
    if policy == "random":
        return random_semantic_continuous_policy(env, obs)
    if policy == "force_multi_covered":
        return force_multi_covered_policy(env, obs)
    if policy == "semantic_sac":
        if agent is None:
            raise ValueError("semantic_sac policy requires a loaded SAC agent")
        return agent.select_action(obs, evaluate=True).astype(np.float32)
    raise ValueError(f"Unsupported policy: {policy}")


def force_multi_covered_policy(env: ContinuousSemanticSAoIEnv, obs: np.ndarray) -> np.ndarray:
    if not env.config.multi_user_association:
        raise ValueError("force_multi_covered policy requires multi_user_association=True")

    action = np.zeros(env.action_dim, dtype=float)
    action[0] = 0.0
    action[1] = 0.0
    action[2] = 0.0
    covered_indices = [idx for idx, ue in enumerate(env.ues) if env._is_covered(ue)]
    if covered_indices:
        action[3:-1] = 0.0
        for idx in covered_indices:
            action[3 + idx] = 1.0
        max_aoi = max(float(np.max(env.aoi)), 1e-6)
        covered_aois = [float(env.aoi[idx]) for idx in covered_indices]
        action[-1] = float(np.clip(max(covered_aois) / max_aoi, 0.0, 1.0))
    else:
        action[3:-1] = 0.0
        action[-1] = 1.0
    return action


ENERGY_STATE_TO_CODE = {
    "normal": 0,
    "return": 1,
    "charging": 2,
    "resume": 3,
    "depleted": 4,
}


def rollout_trace(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    config = ContinuousSemanticSAoIEnvConfig(
        max_steps=args.steps,
        num_ues=args.num_ues,
        seed=args.seed,
        service_radius=args.service_radius,
        move_speed=args.move_speed,
        resource_allocation_mode=args.resource_allocation_mode,
        scheduler_mode=args.scheduler_mode,
        multi_user_association=args.multi_user_association,
        association_threshold=args.association_threshold,
        energy_max=args.energy_max,
        energy_return_threshold=args.energy_return_threshold,
        energy_recovery_threshold=args.energy_recovery_threshold,
        energy_hap_power=args.energy_hap_power,
    )
    env = ContinuousSemanticSAoIEnv(config)
    obs = env.reset(seed=args.seed)
    agent = build_semantic_sac_agent(env, args.semantic_sac_model) if args.policy == "semantic_sac" else None

    bandwidth_trace: List[np.ndarray] = []
    power_trace: List[np.ndarray] = []
    saoi_trace: List[np.ndarray] = []
    diagnostics: List[Dict] = []

    for _ in range(args.steps):
        action = select_action(env, obs, args.policy, agent=agent)
        obs, _, done, info = env.step(action)
        bandwidth_trace.append(np.asarray(info["bandwidth_alloc"], dtype=float))
        power_trace.append(np.asarray(info["power_alloc"], dtype=float))
        saoi_trace.append(np.asarray(info["saoi"], dtype=float))
        diagnostics.append(
            {
                "energy_state": str(info.get("energy_state", "unknown")),
                "energy": float(info.get("energy", 0.0)),
                "covered_ues_count": int(info.get("covered_ues", 0)),
                "selected_user_count": int(info.get("selected_user_count", 1 if info.get("selected_ue") is not None else 0)),
                "success_count": int(info.get("success_count", 1 if info.get("success") else 0)),
                "bit_success_count": int(info.get("bit_success_count", 1 if info.get("bit_success") else 0)),
                "selected_ues": list(info.get("selected_ues", [info["selected_ue"]] if info.get("selected_ue") is not None else [])),
                "selected_target_indices": list(info.get("selected_target_indices", [info.get("selected_target_idx", -1)] if info.get("selected_target_idx", -1) >= 0 else [])),
                "per_user_rates": dict(info.get("per_user_rates", {})),
                "association_scores": np.asarray(info.get("action_association_scores", np.array([], dtype=float)), dtype=float).copy(),
            }
        )
        if done:
            break

    bandwidth = np.vstack(bandwidth_trace).T if bandwidth_trace else np.zeros((args.num_ues, 0), dtype=float)
    power = np.vstack(power_trace).T if power_trace else np.zeros((args.num_ues, 0), dtype=float)
    saoi = np.vstack(saoi_trace).T if saoi_trace else np.zeros((args.num_ues, 0), dtype=float)
    return bandwidth, power, saoi, diagnostics


def save_trace_csv(
    output_dir: Path,
    bandwidth: np.ndarray,
    power: np.ndarray,
    saoi: np.ndarray,
    diagnostics: List[Dict],
) -> Path:
    csv_path = ensure_unique_path(output_dir, "resource_allocation_trace", ".csv")
    num_steps = bandwidth.shape[1]
    num_users = bandwidth.shape[0]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "step",
            "energy_state",
            "energy_j",
            "covered_ues_count",
            "selected_user_count",
            "success_count",
            "bit_success_count",
            "selected_ues",
            "selected_target_indices",
        ]
        for user_idx in range(num_users):
            uid = user_idx + 1
            fieldnames.extend(
                [
                    f"user_{uid}_association_score",
                    f"user_{uid}_bandwidth_mhz",
                    f"user_{uid}_power_mw",
                    f"user_{uid}_rate_mbps",
                    f"user_{uid}_saoi",
                ]
            )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for step_idx in range(num_steps):
            diagnostic = diagnostics[step_idx] if step_idx < len(diagnostics) else {}
            association_scores = np.asarray(diagnostic.get("association_scores", np.array([], dtype=float)), dtype=float)
            per_user_rates = diagnostic.get("per_user_rates", {})
            row = {
                "step": step_idx + 1,
                "energy_state": diagnostic.get("energy_state", "unknown"),
                "energy_j": diagnostic.get("energy", 0.0),
                "covered_ues_count": diagnostic.get("covered_ues_count", 0),
                "selected_user_count": diagnostic.get("selected_user_count", 0),
                "success_count": diagnostic.get("success_count", 0),
                "bit_success_count": diagnostic.get("bit_success_count", 0),
                "selected_ues": ",".join(str(uid) for uid in diagnostic.get("selected_ues", [])),
                "selected_target_indices": ",".join(str(idx) for idx in diagnostic.get("selected_target_indices", [])),
            }
            for user_idx in range(num_users):
                uid = user_idx + 1
                assoc_score = float(association_scores[user_idx]) if user_idx < association_scores.size else 0.0
                row[f"user_{uid}_bandwidth_mhz"] = bandwidth[user_idx, step_idx] / 1e6
                row[f"user_{uid}_power_mw"] = power[user_idx, step_idx] * 1e3
                row[f"user_{uid}_association_score"] = assoc_score
                row[f"user_{uid}_rate_mbps"] = float(per_user_rates.get(uid, 0.0)) / 1e6
                row[f"user_{uid}_saoi"] = saoi[user_idx, step_idx]
            writer.writerow(row)
    return csv_path


def plot_heatmap(data: np.ndarray, title: str, colorbar_label: str, output_dir: Path, prefix: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(data, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Time slot index")
    ax.set_ylabel("User index")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([f"UE {idx + 1}" for idx in range(data.shape[0])])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path = ensure_unique_path(output_dir, prefix, ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_saoi(saoi: np.ndarray, output_dir: Path) -> Path:
    steps = np.arange(1, saoi.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for user_idx in range(saoi.shape[0]):
        ax.plot(steps, saoi[user_idx], linewidth=1.8, label=f"UE {user_idx + 1}")
    if saoi.size:
        ax.plot(steps, np.mean(saoi, axis=0), linestyle="--", linewidth=2.2, color="black", label="Mean SAoI")
    ax.set_title("Per-user SAoI Evolution")
    ax.set_xlabel("Time slot index")
    ax.set_ylabel("SAoI (time slots)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = ensure_unique_path(output_dir, "saoi_per_user", ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_uav_state(diagnostics: List[Dict], output_dir: Path) -> Path:
    steps = np.arange(1, len(diagnostics) + 1)
    state_codes = np.array(
        [ENERGY_STATE_TO_CODE.get(str(item.get("energy_state", "unknown")), -1) for item in diagnostics],
        dtype=float,
    )
    energy_values = np.array([float(item.get("energy", 0.0)) for item in diagnostics], dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.step(steps, state_codes, where="mid", color="tab:red", linewidth=2, label="UAV energy state")
    ax1.set_xlabel("Time slot index")
    ax1.set_ylabel("UAV state code")
    ax1.set_yticks(list(ENERGY_STATE_TO_CODE.values()))
    ax1.set_yticklabels(list(ENERGY_STATE_TO_CODE.keys()))
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(steps, energy_values, color="tab:blue", linewidth=1.8, label="Remaining energy")
    ax2.set_ylabel("Energy (J)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("UAV Energy State and Remaining Energy")
    fig.tight_layout()
    path = ensure_unique_path(output_dir, "uav_state_trace", ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_user_activity(diagnostics: List[Dict], output_dir: Path) -> Path:
    steps = np.arange(1, len(diagnostics) + 1)
    covered_counts = np.array([int(item.get("covered_ues_count", 0)) for item in diagnostics], dtype=float)
    selected_counts = np.array([int(item.get("selected_user_count", 0)) for item in diagnostics], dtype=float)
    success_counts = np.array([int(item.get("success_count", 0)) for item in diagnostics], dtype=float)
    bit_success_counts = np.array([int(item.get("bit_success_count", 0)) for item in diagnostics], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(steps, covered_counts, linewidth=2.0, color="tab:blue", label="Covered users")
    ax.plot(steps, selected_counts, linewidth=2.0, color="tab:orange", label="Associated/active users")
    ax.plot(steps, success_counts, linewidth=1.8, color="tab:green", label="Semantic-success users")
    ax.plot(steps, bit_success_counts, linewidth=1.8, color="tab:red", linestyle="--", label="Bit-success users")
    ax.set_title("User Coverage, Association and Success Counts")
    ax.set_xlabel("Time slot index")
    ax.set_ylabel("User count")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = ensure_unique_path(output_dir, "user_activity_trace", ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot bandwidth/power allocation and per-user SAoI traces")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default="rule", choices=["rule", "random", "force_multi_covered", "semantic_sac"])
    parser.add_argument("--resource-allocation-mode", type=str, default="uniform", choices=["fixed", "uniform"])
    parser.add_argument("--scheduler-mode", type=str, default="aoi_weighted", choices=["round_robin", "equal_share", "aoi_weighted"])
    parser.add_argument("--service-radius", type=float, default=180.0)
    parser.add_argument("--move-speed", type=float, default=10.0)
    parser.add_argument("--energy-max", type=float, default=8000.0)
    parser.add_argument("--energy-return-threshold", type=float, default=0.12)
    parser.add_argument("--energy-recovery-threshold", type=float, default=0.6)
    parser.add_argument("--energy-hap-power", type=float, default=450.0)
    parser.add_argument("--multi-user-association", action="store_true")
    parser.add_argument("--association-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-sac-model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/resource_plots")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.policy == "semantic_sac" and not args.semantic_sac_model:
        raise ValueError("--semantic-sac-model is required when --policy semantic_sac is used")
    output_dir = Path(args.output_dir)

    bandwidth, power, saoi, diagnostics = rollout_trace(args)
    csv_path = save_trace_csv(output_dir, bandwidth, power, saoi, diagnostics)
    print(f"saved_csv(resource_allocation_trace)={csv_path}")

    bandwidth_path = plot_heatmap(
        bandwidth / 1e6,
        title="Bandwidth Allocation per User",
        colorbar_label="Bandwidth (MHz)",
        output_dir=output_dir,
        prefix="bandwidth_allocation",
    )
    print(f"saved_plot(bandwidth_allocation)={bandwidth_path}")

    power_path = plot_heatmap(
        power * 1e3,
        title="Transmit Power Allocation per User",
        colorbar_label="Power (mW)",
        output_dir=output_dir,
        prefix="power_allocation",
    )
    print(f"saved_plot(power_allocation)={power_path}")

    saoi_path = plot_saoi(saoi, output_dir=output_dir)
    print(f"saved_plot(saoi_per_user)={saoi_path}")

    uav_state_path = plot_uav_state(diagnostics, output_dir=output_dir)
    print(f"saved_plot(uav_state_trace)={uav_state_path}")

    user_activity_path = plot_user_activity(diagnostics, output_dir=output_dir)
    print(f"saved_plot(user_activity_trace)={user_activity_path}")


if __name__ == "__main__":
    main()
