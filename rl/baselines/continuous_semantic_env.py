from __future__ import annotations

from typing import Dict

import numpy as np

from rl.envs import ContinuousSemanticSAoIEnv, ContinuousSemanticSAoIEnvConfig


def random_semantic_continuous_policy(env: ContinuousSemanticSAoIEnv, obs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            env.rng.uniform(-1.0, 1.0),
            env.rng.uniform(-1.0, 1.0),
            env.rng.uniform(0.0, 1.0),
            env.rng.uniform(0.0, 1.0),
            env.rng.uniform(0.0, 1.0),
        ],
        dtype=float,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def continuous_rule_semantic_policy(env: ContinuousSemanticSAoIEnv, obs: np.ndarray) -> np.ndarray:
    scores = []
    for idx, ue in enumerate(env.ues):
        horizontal_distance = float(np.linalg.norm(ue.position - env.uav.position[:2]))
        scores.append(float(env.aoi[idx] / (horizontal_distance + 25.0)))
    target_idx = int(np.argmax(scores))
    target_ue = env.ues[target_idx]
    target_vec = np.array(
        [target_ue.position[0] - env.uav.position[0], target_ue.position[1] - env.uav.position[1]],
        dtype=float,
    )
    move_dir = _normalize(target_vec)
    queue_ratio = env.virtual_energy_queue / max(env.energy_model.E_max, 1.0)
    energy_ratio = env.uav.energy / max(env.energy_model.E_max, 1.0)
    charge_bias = 0.0
    if queue_ratio >= 0.55:
        charge_bias = 0.8
    elif queue_ratio >= 0.30:
        charge_bias = 0.45
    elif energy_ratio <= 0.35:
        charge_bias = 0.35

    # Higher AoI -> prefer higher compression ratio (better quality)
    max_aoi = max(float(np.max(env.aoi)), 1e-6)
    ratio_action = float(np.clip(env.aoi[target_idx] / max_aoi, 0.0, 1.0))

    return np.array([move_dir[0], move_dir[1], charge_bias, (target_idx + 0.5) / env.config.num_ues, ratio_action], dtype=float)


def run_semantic_policy_episode(
    policy_name: str,
    config: ContinuousSemanticSAoIEnvConfig | None = None,
    seed: int = 42,
) -> Dict[str, float]:
    env = ContinuousSemanticSAoIEnv(config or ContinuousSemanticSAoIEnvConfig(seed=seed))
    obs = env.reset(seed=seed)

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
        if policy_name == "random_semantic_continuous":
            action = random_semantic_continuous_policy(env, obs)
        elif policy_name == "continuous_rule_semantic":
            action = continuous_rule_semantic_policy(env, obs)
        else:
            raise ValueError(f"Unsupported semantic continuous policy: {policy_name}")

        obs, reward, done, info = env.step(action)
        total_reward += reward
        mean_saois.append(info["mean_saoi"])
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
        "policy": policy_name,
        "episode_reward": total_reward,
        "avg_mean_saoi": float(np.mean(mean_saois)) if mean_saois else 0.0,
        "avg_mean_aoi": float(np.mean(mean_aois)) if mean_aois else 0.0,
        "final_energy": float(env.uav.energy),
        "final_state": env.uav.energy_state,
        "charge_steps": charge_steps,
        "success_updates": success_updates,
        "service_attempts": service_attempts,
        "success_rate": success_updates / max(service_attempts, 1),
        "avg_queue": float(np.mean(queue_values)) if queue_values else 0.0,
        "final_queue": float(queue_values[-1]) if queue_values else 0.0,
        "max_queue": max_queue,
    }
