from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from rl.envs.aoi_energy_env import AoIEnvConfig, SingleUAVAoIEnv
from semantic_jscc import DeepJSCCScenarioModule


@dataclass
class SAoIEnvConfig(AoIEnvConfig):
    checkpoint_path: str = "semantic_jscc/checkpoints/deepjscc_best.pth"
    semantic_quality_weight: float = 0.7
    semantic_utility_weight: float = 0.3
    semantic_utility_scale: float = 5.0
    semantic_min_gain: float = 0.1
    semantic_reset_threshold: float = 0.85
    semantic_image_size: int = 32
    semantic_compression_ratio: float = 0.5
    semantic_cost_weight: float = 0.1
    semantic_mask_mode: str = "uniform"


class SingleUAVSAoIEnv(SingleUAVAoIEnv):
    """单 UAV、SAoI、李雅普诺夫能量队列离散动作环境。"""

    def __init__(self, config: SAoIEnvConfig | None = None):
        super().__init__(config or SAoIEnvConfig())
        self.config: SAoIEnvConfig
        self.saoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.semantic_scores = np.zeros(self.config.num_ues, dtype=float)
        self.semantic_module = DeepJSCCScenarioModule(
            checkpoint_path=self.config.checkpoint_path,
            mask_mode=self.config.semantic_mask_mode,
            strict_checkpoint=False,
        )
        self.semantic_images = self._build_semantic_image_bank()

    def _build_semantic_image_bank(self) -> list[np.ndarray]:
        image_rng = np.random.default_rng(self.config.seed + 2026)
        size = self.config.semantic_image_size
        return [image_rng.random((3, size, size), dtype=np.float32) for _ in range(self.config.num_ues)]

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self.saoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.semantic_scores = np.zeros(self.config.num_ues, dtype=float)
        return obs

    def _get_link_sinr_db(self, selected_ue) -> float:
        results = self.a2g_channel.compute_sinr_and_rate(self.uav, [selected_ue], [self.uav])
        if not results:
            return -10.0
        sinr_linear = float(results[0]["sinr"])
        return float(10.0 * np.log10(max(sinr_linear, 1e-10)))

    def _semantic_freshness_gain(self, semantic_quality: float, semantic_utility: float) -> float:
        utility_score = 1.0 / (1.0 + np.exp(-self.config.semantic_utility_scale * semantic_utility))
        gain = (
            self.config.semantic_quality_weight * semantic_quality
            + self.config.semantic_utility_weight * float(utility_score)
        )
        return float(np.clip(gain, self.config.semantic_min_gain, 1.0))

    def _update_saoi(
        self,
        selected_ue,
        success: bool,
        semantic_quality: float,
        semantic_utility: float,
    ) -> None:
        self.saoi += self.config.delta_t
        if selected_ue is None or not success:
            return

        idx = selected_ue.uid - 1
        gain = self._semantic_freshness_gain(semantic_quality=semantic_quality, semantic_utility=semantic_utility)
        self.semantic_scores[idx] = semantic_quality

        if semantic_quality >= self.config.semantic_reset_threshold:
            self.saoi[idx] = self.config.delta_t
            return

        updated = (1.0 - gain) * self.saoi[idx] + self.config.delta_t
        self.saoi[idx] = max(self.config.delta_t, float(updated))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, Dict]:
        charge_flag, movement_idx, service_idx = self.decode_action(int(action))
        forced_return = False
        if self.config.energy_hard_constraint:
            charge_flag, movement_idx, service_idx, forced_return = self._enforce_energy_hard_constraint(
                charge_flag,
                movement_idx,
                service_idx,
            )

        prev_queue = self.virtual_energy_queue
        prev_mean_saoi = float(np.mean(self.saoi))

        self._apply_action(charge_flag, movement_idx)
        self._advance_entities()

        selected_ue = None if service_idx == 0 else self.ues[service_idx - 1]
        rate = 0.0
        success = False
        semantic_quality = 0.0
        semantic_utility = 0.0
        semantic_cost = 0.0

        if selected_ue is not None and self.uav.energy_state == "normal":
            rate = self._get_link_rate(selected_ue)
            required_rate = self.config.packet_size_bits / self.config.delta_t
            success = rate >= required_rate

            if success:
                semantic_result = self.semantic_module.transmit(
                    image=self.semantic_images[selected_ue.uid - 1],
                    sinr_db=self._get_link_sinr_db(selected_ue),
                    compression_ratio=self.config.semantic_compression_ratio,
                    task_weights={
                        "quality": 1.0,
                        "cost": self.config.semantic_cost_weight,
                    },
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
            "selected_ue": None if selected_ue is None else selected_ue.uid,
            "success": success,
            "rate": rate,
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "queue_norm": float(queue_norm),
            "drift": float(drift),
            "saoi_improvement": float(saoi_improvement),
            "forced_return": forced_return,
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info
