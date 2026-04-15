from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.aoi_energy_env import AoIEnvConfig, SingleUAVAoIEnv


def random_policy(env: SingleUAVAoIEnv, obs: np.ndarray) -> int:
    return int(env.rng.integers(0, env.num_actions))


def heuristic_policy(env: SingleUAVAoIEnv, obs: np.ndarray) -> int:
    service_size = env.config.num_ues + 1
    charge_flag = 0
    movement_idx = env.MOVEMENTS.index("hover")
    service_idx = 0

    if env.uav.energy_state in {"return", "charging", "resume"}:
        charge_flag = 1
        return charge_flag * len(env.MOVEMENTS) * service_size + movement_idx * service_size + service_idx

    dynamic_return_threshold = max(
        env.energy_model.return_threshold * env.energy_model.E_max,
        env._estimate_return_energy_budget(),
    )
    if env.uav.energy <= dynamic_return_threshold:
        charge_flag = 1
        return charge_flag * len(env.MOVEMENTS) * service_size + movement_idx * service_size + service_idx

    target_idx = int(np.argmax(env.aoi))
    target_ue = env.ues[target_idx]
    service_idx = target_idx + 1 if env._is_covered(target_ue) else 0

    dx = target_ue.position[0] - env.uav.position[0]
    dy = target_ue.position[1] - env.uav.position[1]
    if abs(dx) > abs(dy):
        movement_idx = env.MOVEMENTS.index("east" if dx > 0 else "west")
    else:
        movement_idx = env.MOVEMENTS.index("north" if dy > 0 else "south")

    if env._is_covered(target_ue):
        movement_idx = env.MOVEMENTS.index("hover")

    return charge_flag * len(env.MOVEMENTS) * service_size + movement_idx * service_size + service_idx


def run_policy_episode(
    policy_name: str,
    config: AoIEnvConfig | None = None,
    seed: int = 42,
) -> Dict[str, float]:
    env = SingleUAVAoIEnv(config or AoIEnvConfig(seed=seed))
    obs = env.reset(seed=seed)

    total_reward = 0.0
    mean_aois = []
    charge_steps = 0
    success_updates = 0
    service_attempts = 0
    max_queue = 0.0

    done = False
    while not done:
        if policy_name == "random":
            action = random_policy(env, obs)
        elif policy_name == "heuristic":
            action = heuristic_policy(env, obs)
        else:
            raise ValueError(f"Unsupported policy: {policy_name}")
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


def run_heuristic_episode(config: AoIEnvConfig | None = None, seed: int = 42) -> Dict[str, float]:
    return run_policy_episode(policy_name="heuristic", config=config, seed=seed)


if __name__ == "__main__":
    summary = run_heuristic_episode()
    for key, value in summary.items():
        print(f"{key}: {value}")
