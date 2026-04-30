from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from semantic_jscc import DeepJSCCScenarioModule

from .ofdma_aoi_energy_env import OFDMAAoIEnvConfig, SingleUAVOFDMAAoIEnv


@dataclass
class ContinuousSemanticBaseSAoIEnvConfig(OFDMAAoIEnvConfig):
    action_dim: int = 4
    charge_bias_threshold: float = 0.55
    service_distance_epsilon: float = 25.0
    compression_ratio_min: float = 0.25
    compression_ratio_max: float = 1.0
    checkpoint_path: str = "semantic_jscc/checkpoints/deepjscc_best.pth"
    semantic_quality_weight: float = 0.7
    semantic_utility_weight: float = 0.3
    semantic_utility_scale: float = 5.0
    semantic_min_gain: float = 0.1
    semantic_reset_threshold: float = 0.85
    semantic_image_size: float = 32
    semantic_cost_weight: float = 0.1
    semantic_mask_mode: str = "uniform"


class ContinuousSemanticBaseSAoIEnv(SingleUAVOFDMAAoIEnv):
    """Continuous SAoI env with semantic compression action but heuristic user selection."""

    def __init__(self, config: ContinuousSemanticBaseSAoIEnvConfig | None = None):
        super().__init__(config or ContinuousSemanticBaseSAoIEnvConfig())
        self.config: ContinuousSemanticBaseSAoIEnvConfig
        self.action_dim = self.config.action_dim
        self.action_low = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.saoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.semantic_scores = np.zeros(self.config.num_ues, dtype=float)
        self.semantic_module = DeepJSCCScenarioModule(
            checkpoint_path=self.config.checkpoint_path,
            mask_mode=self.config.semantic_mask_mode,
            strict_checkpoint=False,
        )
        self.semantic_images = self._build_semantic_image_bank()

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.zeros_like(vec)
        return vec / norm

    def _build_semantic_image_bank(self) -> list[np.ndarray]:
        image_rng = np.random.default_rng(self.config.seed + 2026)
        size = int(self.config.semantic_image_size)
        return [image_rng.random((3, size, size), dtype=np.float32) for _ in range(self.config.num_ues)]

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self.saoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.semantic_scores = np.zeros(self.config.num_ues, dtype=float)
        return obs

    def _select_service_target(self) -> int:
        scores = []
        for idx, ue in enumerate(self.ues):
            horizontal_distance = float(np.linalg.norm(ue.position - self.uav.position[:2]))
            scores.append(float(self.aoi[idx] / (horizontal_distance + self.config.service_distance_epsilon)))
        return int(np.argmax(scores))

    def _build_task_direction(self, action_xy: np.ndarray) -> np.ndarray:
        direction = np.array([float(action_xy[0]), float(action_xy[1]), 0.0], dtype=float)
        return self._normalize(direction)

    def _build_charge_direction(self) -> np.ndarray:
        return self._normalize(self._get_charging_waypoint() - self.uav.position)

    def _decode_compression_ratio(self, action_ratio: float) -> float:
        clipped = float(np.clip(action_ratio, 0.0, 1.0))
        return float(
            self.config.compression_ratio_min
            + clipped * (self.config.compression_ratio_max - self.config.compression_ratio_min)
        )

    def _semantic_freshness_gain(self, semantic_quality: float, semantic_utility: float) -> float:
        utility_score = 1.0 / (1.0 + np.exp(-self.config.semantic_utility_scale * semantic_utility))
        gain = (
            self.config.semantic_quality_weight * semantic_quality
            + self.config.semantic_utility_weight * float(utility_score)
        )
        return float(np.clip(gain, self.config.semantic_min_gain, 1.0))

    def _update_saoi(self, selected_ue, success: bool, semantic_quality: float, semantic_utility: float) -> None:
        self.saoi += self.config.delta_t
        if selected_ue is None or not success:
            return
        idx = selected_ue.uid - 1
        gain = self._semantic_freshness_gain(semantic_quality, semantic_utility)
        self.semantic_scores[idx] = semantic_quality
        if semantic_quality >= self.config.semantic_reset_threshold:
            self.saoi[idx] = self.config.delta_t
            return
        updated = (1.0 - gain) * self.saoi[idx] + self.config.delta_t
        self.saoi[idx] = max(self.config.delta_t, float(updated))

    def _apply_continuous_action(self, action: np.ndarray) -> int:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"continuous action must have shape ({self.action_dim},)")

        if self.uav.energy_state in {"charging", "resume", "depleted", "return"}:
            return 1

        a_x = float(np.clip(action[0], -1.0, 1.0))
        a_y = float(np.clip(action[1], -1.0, 1.0))
        a_c = float(np.clip(action[2], 0.0, 1.0))

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
        return charge_flag

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict]:
        action = np.asarray(action, dtype=float).reshape(-1)
        prev_queue = self.virtual_energy_queue
        prev_mean_saoi = float(np.mean(self.saoi))

        charge_flag = self._apply_continuous_action(action)
        compression_ratio = self._decode_compression_ratio(action[3])
        self._advance_entities()

        target_idx = self._select_service_target()
        service_idx = target_idx + 1 if self._is_covered(self.ues[target_idx]) and charge_flag == 0 else 0
        selected_ue = None if service_idx == 0 else self.ues[service_idx - 1]
        link_info = self._get_ofdma_link_info(selected_ue)
        rate = float(link_info["rate"])
        success = False
        semantic_attempt = False
        semantic_quality = 0.0
        semantic_utility = 0.0
        semantic_cost = 0.0

        if selected_ue is not None and self.uav.energy_state == "normal":
            semantic_attempt = True
            required_rate = self.config.packet_size_bits / self.config.delta_t
            success = rate >= required_rate
            if success:
                mean_sinr = max(float(link_info["mean_sinr"]), 1e-10)
                semantic_result = self.semantic_module.transmit(
                    image=self.semantic_images[selected_ue.uid - 1],
                    sinr_db=float(10.0 * np.log10(mean_sinr)),
                    compression_ratio=compression_ratio,
                    task_weights={"quality": 1.0, "cost": self.config.semantic_cost_weight},
                )
                semantic_quality = semantic_result.semantic_quality
                semantic_utility = semantic_result.semantic_utility
                semantic_cost = semantic_result.transmission_cost

        self._update_aoi(selected_ue, success)
        self._update_saoi(selected_ue, success, semantic_quality, semantic_utility)
        energy_info = self._update_energy(selected_ue, charge_flag)

        mean_saoi = float(np.mean(self.saoi))
        self.virtual_energy_queue = self._update_virtual_energy_queue(prev_queue, energy_info)
        drift = 0.5 * (self.virtual_energy_queue**2 - prev_queue**2) / (self.energy_model.E_max**2)
        mean_saoi_norm = mean_saoi / (self.config.max_steps * self.config.delta_t)
        queue_norm = self.virtual_energy_queue / max(self.energy_model.E_max, 1.0)
        saoi_improvement = (prev_mean_saoi - mean_saoi) / max(self.config.delta_t, 1.0)
        reward = -(
            self.config.lyapunov_v * mean_saoi_norm
            + self.config.drift_weight * drift
            + self.config.queue_weight * queue_norm
        )
        reward += saoi_improvement
        if success:
            reward += self.config.success_reward + semantic_utility
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
            "mean_saoi": mean_saoi,
            "saoi": self.saoi.copy(),
            "mean_aoi": float(np.mean(self.aoi)),
            "aoi": self.aoi.copy(),
            "semantic_scores": self.semantic_scores.copy(),
            "semantic_quality": float(semantic_quality),
            "semantic_utility": float(semantic_utility),
            "semantic_cost": float(semantic_cost),
            "semantic_attempt": semantic_attempt,
            "semantic_compression_ratio": compression_ratio,
            "selected_ue": None if selected_ue is None else selected_ue.uid,
            "selected_target_idx": int(target_idx),
            "success": success,
            "rate": rate,
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "queue_norm": float(queue_norm),
            "drift": float(drift),
            "saoi_improvement": float(saoi_improvement),
            "scheduler_mode": self.config.scheduler_mode,
            "assigned_rbs": link_info["assigned_rbs"],
            "mean_sinr": float(link_info["mean_sinr"]),
            "covered_ues": int(link_info["covered_ues"]),
            "action_ax": float(np.clip(action[0], -1.0, 1.0)),
            "action_ay": float(np.clip(action[1], -1.0, 1.0)),
            "action_charge_bias": float(np.clip(action[2], 0.0, 1.0)),
            "action_semantic_ratio": float(np.clip(action[3], 0.0, 1.0)),
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info
