from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from rl.envs.continuous_aoi_energy_env import ContinuousAoIEnvConfig, ContinuousSingleUAVAoIEnv


@dataclass
class ContinuousAssocAoIEnvConfig(ContinuousAoIEnvConfig):
    action_dim: int = 4


class ContinuousAssocSingleUAVAoIEnv(ContinuousSingleUAVAoIEnv):
    """连续动作 + 用户关联版环境。

    动作定义:
    - a_x in [-1, 1]
    - a_y in [-1, 1]
    - a_c in [0, 1]
    - a_u in [0, 1]
    """

    def __init__(self, config: ContinuousAssocAoIEnvConfig | None = None):
        super().__init__(config or ContinuousAssocAoIEnvConfig())
        self.config: ContinuousAssocAoIEnvConfig
        self.action_dim = self.config.action_dim
        self.action_low = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def _decode_service_target(self, action_service: float) -> int:
        clipped = float(np.clip(action_service, 0.0, 1.0 - 1e-8))
        return min(int(clipped * self.config.num_ues), self.config.num_ues - 1)

    def _apply_continuous_action(self, action: np.ndarray) -> tuple[int, int]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"continuous action must have shape ({self.action_dim},)")

        if self.uav.energy_state in {"charging", "resume", "depleted", "return"}:
            return 1, 0

        a_x = float(np.clip(action[0], -1.0, 1.0))
        a_y = float(np.clip(action[1], -1.0, 1.0))
        a_c = float(np.clip(action[2], 0.0, 1.0))
        a_u = float(np.clip(action[3], 0.0, 1.0))

        dynamic_return_threshold = max(
            self.energy_model.return_threshold * self.energy_model.E_max,
            self._estimate_return_energy_budget(),
        )
        force_return = self.uav.energy <= dynamic_return_threshold
        charge_flag = 1 if force_return or a_c >= self.config.charge_bias_threshold else 0

        if charge_flag == 1:
            move_dir = self._build_charge_direction()
        else:
            task_dir = self._build_task_direction(np.array([a_x, a_y], dtype=float))
            charge_dir = self._build_charge_direction()
            move_dir = self._normalize((1.0 - a_c) * task_dir + a_c * charge_dir)

        velocity = np.zeros(3, dtype=float)
        velocity[:2] = self.config.move_speed * move_dir[:2]
        self.uav.velocity = velocity
        if charge_flag == 1 and self.uav.energy_state == "normal":
            self.uav.energy_state = "return"
        elif self.uav.energy_state == "normal":
            self.uav.energy_state = "normal"

        target_idx = self._decode_service_target(a_u)
        return charge_flag, target_idx

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict]:
        prev_queue = self.virtual_energy_queue
        prev_mean_aoi = float(np.mean(self.aoi))

        charge_flag, target_idx = self._apply_continuous_action(action)
        self._advance_entities()

        service_idx = target_idx + 1 if self._is_covered(self.ues[target_idx]) and charge_flag == 0 else 0
        selected_ue = None if service_idx == 0 else self.ues[service_idx - 1]
        rate = 0.0
        success = False
        if selected_ue is not None and self.uav.energy_state == "normal":
            rate = self._get_link_rate(selected_ue)
            required_rate = self.config.packet_size_bits / self.config.delta_t
            success = rate >= required_rate

        self._update_aoi(selected_ue, success)
        energy_info = self._update_energy(selected_ue, charge_flag)

        mean_aoi = float(np.mean(self.aoi))
        self.virtual_energy_queue = self._update_virtual_energy_queue(prev_queue, energy_info)
        drift = 0.5 * (self.virtual_energy_queue**2 - prev_queue**2) / (self.energy_model.E_max**2)
        mean_aoi_norm = mean_aoi / (self.config.max_steps * self.config.delta_t)
        queue_norm = self.virtual_energy_queue / max(self.energy_model.E_max, 1.0)
        aoi_improvement = (prev_mean_aoi - mean_aoi) / max(self.config.delta_t, 1.0)

        reward = -(
            self.config.lyapunov_v * mean_aoi_norm
            + self.config.drift_weight * drift
            + self.config.queue_weight * queue_norm
        )
        reward += aoi_improvement

        if success:
            reward += self.config.success_reward
        elif selected_ue is not None:
            reward -= self.config.invalid_service_penalty

        if self.uav.energy_state in {"return", "charging", "resume"}:
            reward -= self.config.charge_step_penalty

        if self.uav.energy_state == "depleted":
            reward -= self.config.depleted_penalty

        self.current_step += 1
        done = self.current_step >= self.config.max_steps or (
            self.config.terminate_on_depleted and self.uav.energy_state == "depleted"
        )

        info = {
            "mean_aoi": mean_aoi,
            "aoi": self.aoi.copy(),
            "selected_ue": None if selected_ue is None else selected_ue.uid,
            "selected_target_idx": int(target_idx),
            "success": success,
            "rate": rate,
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "queue_norm": float(queue_norm),
            "drift": float(drift),
            "aoi_improvement": float(aoi_improvement),
            "action_ax": float(np.clip(action[0], -1.0, 1.0)),
            "action_ay": float(np.clip(action[1], -1.0, 1.0)),
            "action_charge_bias": float(np.clip(action[2], 0.0, 1.0)),
            "action_service_target": float(np.clip(action[3], 0.0, 1.0)),
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info
