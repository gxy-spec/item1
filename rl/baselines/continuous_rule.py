from __future__ import annotations

import numpy as np

from rl.envs import SingleUAVAoIEnv


QUEUE_HIGH_THRESHOLD = 0.55
QUEUE_MODERATE_THRESHOLD = 0.30
AOI_SERVICE_THRESHOLD = 8.0
OPPORTUNISTIC_SERVICE_THRESHOLD = 10.0
SERVICE_SCORE_DISTANCE_EPS = 25.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _movement_from_direction(env: SingleUAVAoIEnv, direction: np.ndarray) -> int:
    horizontal = direction[:2]
    vertical = direction[2]
    if np.linalg.norm(direction) < 1e-8:
        return env.MOVEMENTS.index("hover")

    if abs(vertical) > max(abs(horizontal[0]), abs(horizontal[1])) and abs(vertical) > 0.15:
        return env.MOVEMENTS.index("up" if vertical > 0 else "down")

    if abs(horizontal[0]) > abs(horizontal[1]):
        return env.MOVEMENTS.index("east" if horizontal[0] > 0 else "west")
    return env.MOVEMENTS.index("north" if horizontal[1] > 0 else "south")


def _action_from_components(env: SingleUAVAoIEnv, charge_flag: int, movement_idx: int, service_idx: int) -> int:
    service_size = env.config.num_ues + 1
    return charge_flag * len(env.MOVEMENTS) * service_size + movement_idx * service_size + service_idx


def _ue_service_score(env: SingleUAVAoIEnv, ue_idx: int) -> float:
    ue = env.ues[ue_idx]
    horizontal_distance = float(np.linalg.norm(ue.position - env.uav.position[:2]))
    return float(env.aoi[ue_idx] / (horizontal_distance + SERVICE_SCORE_DISTANCE_EPS))


def _best_service_target(env: SingleUAVAoIEnv) -> int:
    scores = [_ue_service_score(env, idx) for idx in range(env.config.num_ues)]
    return int(np.argmax(scores))


def _best_covered_target(env: SingleUAVAoIEnv) -> int | None:
    covered_indices = [idx for idx, ue in enumerate(env.ues) if env._is_covered(ue)]
    if not covered_indices:
        return None
    return max(covered_indices, key=lambda idx: env.aoi[idx])


def _build_target_vector(env: SingleUAVAoIEnv, ue_idx: int) -> np.ndarray:
    ue = env.ues[ue_idx]
    return np.array(
        [
            ue.position[0] - env.uav.position[0],
            ue.position[1] - env.uav.position[1],
            0.0,
        ],
        dtype=float,
    )


def continuous_rule_policy(env: SingleUAVAoIEnv, obs: np.ndarray) -> int:
    hover_idx = env.MOVEMENTS.index("hover")

    if env.uav.energy_state in {"charging", "resume", "depleted"}:
        return _action_from_components(env, 0, hover_idx, 0)
    if env.uav.energy_state == "return":
        return _action_from_components(env, 1, hover_idx, 0)

    queue_ratio = env.virtual_energy_queue / max(env.energy_model.E_max, 1.0)
    dynamic_return_threshold = max(
        env.energy_model.return_threshold * env.energy_model.E_max,
        env._estimate_return_energy_budget(),
    )
    energy_critical = env.uav.energy <= dynamic_return_threshold
    queue_high = queue_ratio >= QUEUE_HIGH_THRESHOLD
    queue_moderate = queue_ratio >= QUEUE_MODERATE_THRESHOLD

    main_target_idx = _best_service_target(env)
    covered_target_idx = _best_covered_target(env)
    charge_waypoint = env._get_charging_waypoint()
    charge_vec = charge_waypoint - env.uav.position

    if energy_critical:
        movement_idx = _movement_from_direction(env, charge_vec)
        return _action_from_components(env, 1, movement_idx, 0)

    if queue_high:
        if covered_target_idx is not None and env.aoi[covered_target_idx] >= OPPORTUNISTIC_SERVICE_THRESHOLD:
            return _action_from_components(env, 0, hover_idx, covered_target_idx + 1)

        blended_direction = 0.35 * _normalize(_build_target_vector(env, main_target_idx)) + 0.65 * _normalize(charge_vec)
        movement_idx = _movement_from_direction(env, blended_direction)
        return _action_from_components(env, 0, movement_idx, 0)

    if covered_target_idx is not None and env.aoi[covered_target_idx] >= AOI_SERVICE_THRESHOLD:
        return _action_from_components(env, 0, hover_idx, covered_target_idx + 1)

    target_vec = _build_target_vector(env, main_target_idx)
    if queue_moderate:
        target_direction = 0.75 * _normalize(target_vec) + 0.25 * _normalize(charge_vec)
    else:
        target_direction = _normalize(target_vec)

    movement_idx = _movement_from_direction(env, target_direction)
    service_idx = 0
    if env._is_covered(env.ues[main_target_idx]):
        movement_idx = hover_idx
        service_idx = main_target_idx + 1

    return _action_from_components(env, 0, movement_idx, service_idx)
