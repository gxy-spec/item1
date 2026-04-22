from __future__ import annotations

from typing import Dict

import numpy as np

from rl.envs import ContinuousAoIEnvConfig, ContinuousSingleUAVAoIEnv


QUEUE_HIGH_THRESHOLD = 0.55
QUEUE_MODERATE_THRESHOLD = 0.30
AOI_SERVICE_THRESHOLD = 8.0
SERVICE_SCORE_DISTANCE_EPS = 25.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def random_continuous_policy(env: ContinuousSingleUAVAoIEnv, obs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            env.rng.uniform(-1.0, 1.0),
            env.rng.uniform(-1.0, 1.0),
            env.rng.uniform(0.0, 1.0),
        ],
        dtype=float,
    )


def _ue_service_score(env: ContinuousSingleUAVAoIEnv, ue_idx: int) -> float:
    ue = env.ues[ue_idx]
    horizontal_distance = float(np.linalg.norm(ue.position - env.uav.position[:2]))
    return float(env.aoi[ue_idx] / (horizontal_distance + SERVICE_SCORE_DISTANCE_EPS))


def _best_service_target(env: ContinuousSingleUAVAoIEnv) -> int:
    scores = [_ue_service_score(env, idx) for idx in range(env.config.num_ues)]
    return int(np.argmax(scores))


def continuous_rule_continuous_policy(
    env: ContinuousSingleUAVAoIEnv,
    obs: np.ndarray,
) -> np.ndarray:
    if env.uav.energy_state in {"charging", "resume", "depleted"}:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    target_idx = _best_service_target(env)
    target_ue = env.ues[target_idx]
    target_vec = np.array(
        [
            target_ue.position[0] - env.uav.position[0],
            target_ue.position[1] - env.uav.position[1],
        ],
        dtype=float,
    )
    task_dir = _normalize(target_vec)
    queue_ratio = env.virtual_energy_queue / max(env.energy_model.E_max, 1.0)
    dynamic_return_threshold = max(
        env.energy_model.return_threshold * env.energy_model.E_max,
        env._estimate_return_energy_budget(),
    )
    energy_ratio = env.uav.energy / max(env.energy_model.E_max, 1.0)
    energy_critical = env.uav.energy <= dynamic_return_threshold
    covered_target = env._is_covered(target_ue)
    charge_bias = 0.0

    if energy_critical:
        charge_bias = 1.0
    elif queue_ratio >= QUEUE_HIGH_THRESHOLD:
        charge_bias = 0.85
    elif queue_ratio >= QUEUE_MODERATE_THRESHOLD:
        charge_bias = 0.45
    elif energy_ratio <= 0.35:
        charge_bias = 0.35

    if covered_target and env.aoi[target_idx] >= AOI_SERVICE_THRESHOLD and not energy_critical:
        return np.array([0.0, 0.0, min(charge_bias, 0.2)], dtype=float)

    return np.array([task_dir[0], task_dir[1], charge_bias], dtype=float)


def run_continuous_policy_episode(
    policy_name: str,
    config: ContinuousAoIEnvConfig | None = None,
    seed: int = 42,
) -> Dict[str, float]:
    env = ContinuousSingleUAVAoIEnv(config or ContinuousAoIEnvConfig(seed=seed))
    obs = env.reset(seed=seed)

    total_reward = 0.0
    mean_aois = []
    charge_steps = 0
    success_updates = 0
    service_attempts = 0
    max_queue = 0.0

    done = False
    while not done:
        if policy_name == "random_continuous":
            action = random_continuous_policy(env, obs)
        elif policy_name == "continuous_rule_continuous":
            action = continuous_rule_continuous_policy(env, obs)
        else:
            raise ValueError(f"Unsupported continuous policy: {policy_name}")

        obs, reward, done, info = env.step(action)
        total_reward += reward
        mean_aois.append(info["mean_aoi"])
        if info["energy_state"] in {"return", "charging", "resume"}:
            charge_steps += 1
        if info["success"]:
            success_updates += 1
        if info["selected_ue"] is not None:
            service_attempts += 1
        max_queue = max(max_queue, float(info["virtual_energy_queue"]))

    return {
        "policy": policy_name,
        "episode_reward": total_reward,
        "avg_mean_aoi": float(np.mean(mean_aois)) if mean_aois else 0.0,
        "final_energy": float(env.uav.energy),
        "final_state": env.uav.energy_state,
        "charge_steps": charge_steps,
        "success_updates": success_updates,
        "service_attempts": service_attempts,
        "success_rate": success_updates / max(service_attempts, 1),
        "max_queue": max_queue,
    }
