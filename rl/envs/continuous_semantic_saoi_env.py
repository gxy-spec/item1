from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from semantic_jscc import DeepJSCCScenarioModule

from .ofdma_aoi_energy_env import OFDMAAoIEnvConfig, SingleUAVOFDMAAoIEnv


@dataclass
class ContinuousSemanticSAoIEnvConfig(OFDMAAoIEnvConfig):
    action_dim: int = 5
    charge_bias_threshold: float = 0.55
    multi_user_association: bool = False
    association_threshold: float = 0.5
    fallback_top1_association: bool = True
    max_associated_users: int | None = None
    compression_ratio_min: float = 0.25
    compression_ratio_max: float = 1.0
    checkpoint_path: str = "semantic_jscc/checkpoints/deepjscc_best.pth"
    semantic_quality_weight: float = 0.7
    semantic_utility_weight: float = 0.3
    semantic_utility_scale: float = 5.0
    semantic_min_gain: float = 0.1
    semantic_reset_threshold: float = 0.85
    semantic_image_size: int = 32
    semantic_cost_weight: float = 0.1
    semantic_mask_mode: str = "uniform"
    semantic_soft_success: bool = True
    semantic_packet_scale: float = 1.0


class ContinuousSemanticSAoIEnv(SingleUAVOFDMAAoIEnv):
    """Continuous OFDMA + SAoI env for semantic SAC.

    Single-user mode:
      action = [a_x, a_y, a_c, a_u, a_r]
    Multi-user mode:
      action = [a_x, a_y, a_c, a_u_1, ..., a_u_N, a_r]
    """

    def __init__(self, config: ContinuousSemanticSAoIEnvConfig | None = None):
        super().__init__(config or ContinuousSemanticSAoIEnvConfig())
        self.config: ContinuousSemanticSAoIEnvConfig
        if self.config.multi_user_association:
            self.action_dim = 4 + self.config.num_ues
            self.action_low = np.array([-1.0, -1.0, 0.0] + [0.0] * self.config.num_ues + [0.0], dtype=np.float32)
            self.action_high = np.array([1.0, 1.0, 1.0] + [1.0] * self.config.num_ues + [1.0], dtype=np.float32)
        else:
            self.action_dim = self.config.action_dim
            self.action_low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

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
        size = self.config.semantic_image_size
        return [image_rng.random((3, size, size), dtype=np.float32) for _ in range(self.config.num_ues)]

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self.saoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.semantic_scores = np.zeros(self.config.num_ues, dtype=float)
        return obs

    def _decode_service_target(self, action_service: float) -> int:
        clipped = float(np.clip(action_service, 0.0, 1.0 - 1e-8))
        return min(int(clipped * self.config.num_ues), self.config.num_ues - 1)

    def _decode_associated_targets(self, association_actions: np.ndarray) -> List[int]:
        scores = np.clip(np.asarray(association_actions, dtype=float).reshape(-1), 0.0, 1.0)
        if scores.size != self.config.num_ues:
            raise ValueError(f"association action must have shape ({self.config.num_ues},)")

        selected = [idx for idx, score in enumerate(scores) if float(score) >= self.config.association_threshold]
        if not selected and self.config.fallback_top1_association and scores.size > 0:
            selected = [int(np.argmax(scores))]

        if self.config.max_associated_users is not None and len(selected) > self.config.max_associated_users:
            ranked = sorted(selected, key=lambda idx: float(scores[idx]), reverse=True)
            selected = ranked[: self.config.max_associated_users]
        return selected

    def _decode_compression_ratio(self, action_ratio: float) -> float:
        clipped = float(np.clip(action_ratio, 0.0, 1.0))
        return float(
            self.config.compression_ratio_min
            + clipped * (self.config.compression_ratio_max - self.config.compression_ratio_min)
        )

    def _required_bit_rate(self) -> float:
        return float(self.config.packet_size_bits / self.config.delta_t)

    def _required_semantic_rate(self, compression_ratio: float) -> float:
        if not self.config.semantic_soft_success:
            return self._required_bit_rate()
        semantic_bits = self.config.packet_size_bits * self.config.semantic_packet_scale * compression_ratio
        return float(semantic_bits / self.config.delta_t)

    def _build_task_direction(self, action_xy: np.ndarray) -> np.ndarray:
        direction = np.array([float(action_xy[0]), float(action_xy[1]), 0.0], dtype=float)
        return self._normalize(direction)

    def _build_charge_direction(self) -> np.ndarray:
        return self._normalize(self._get_charging_waypoint() - self.uav.position)

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
        gain = self._semantic_freshness_gain(
            semantic_quality=semantic_quality,
            semantic_utility=semantic_utility,
        )
        self.semantic_scores[idx] = semantic_quality

        if semantic_quality >= self.config.semantic_reset_threshold:
            self.saoi[idx] = self.config.delta_t
            return

        updated = (1.0 - gain) * self.saoi[idx] + self.config.delta_t
        self.saoi[idx] = max(self.config.delta_t, float(updated))

    def _update_aoi_multi(self, successful_ues: List) -> None:
        self.aoi += self.config.delta_t
        for ue in successful_ues:
            self.aoi[ue.uid - 1] = self.config.delta_t

    def _update_saoi_multi(self, successful_results: List[tuple]) -> None:
        self.saoi += self.config.delta_t
        for ue, semantic_quality, semantic_utility in successful_results:
            idx = ue.uid - 1
            gain = self._semantic_freshness_gain(
                semantic_quality=semantic_quality,
                semantic_utility=semantic_utility,
            )
            self.semantic_scores[idx] = semantic_quality
            if semantic_quality >= self.config.semantic_reset_threshold:
                self.saoi[idx] = self.config.delta_t
                continue
            updated = (1.0 - gain) * self.saoi[idx] + self.config.delta_t
            self.saoi[idx] = max(self.config.delta_t, float(updated))

    def _apply_continuous_action(self, action: np.ndarray) -> tuple[int, List[int]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"continuous action must have shape ({self.action_dim},)")

        if self.uav.energy_state in {"charging", "resume", "depleted", "return"}:
            return 1, []

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

        if self.config.multi_user_association:
            target_indices = self._decode_associated_targets(action[3:-1])
        else:
            a_u = float(np.clip(action[3], 0.0, 1.0))
            target_indices = [self._decode_service_target(a_u)]
        return charge_flag, target_indices

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict]:
        action = np.asarray(action, dtype=float).reshape(-1)
        prev_queue = self.virtual_energy_queue
        prev_mean_saoi = float(np.mean(self.saoi))

        charge_flag, target_indices = self._apply_continuous_action(action)
        compression_ratio = self._decode_compression_ratio(action[-1])
        self._advance_entities()

        selected_ues = []
        if charge_flag == 0:
            selected_ues = [self.ues[idx] for idx in target_indices if self._is_covered(self.ues[idx])]

        link_info = self._get_ofdma_multiuser_link_info(selected_ues)
        per_user = link_info.get("per_user", {})
        bit_required_rate = self._required_bit_rate()
        semantic_required_rate = self._required_semantic_rate(compression_ratio)
        bit_successful_ues = []
        successful_ues = []
        successful_results = []
        semantic_attempt_count = len(selected_ues) if self.uav.energy_state == "normal" else 0
        semantic_qualities = []
        semantic_utilities = []
        semantic_costs = []
        selected_rates = []
        selected_sinrs = []

        for ue in selected_ues:
            metrics = per_user.get(ue.uid, {})
            rate = float(metrics.get("rate", 0.0))
            mean_sinr = float(metrics.get("mean_sinr", 0.0))
            selected_rates.append(rate)
            selected_sinrs.append(mean_sinr)
            bit_success = rate >= bit_required_rate
            semantic_success = rate >= semantic_required_rate
            if bit_success:
                bit_successful_ues.append(ue)
            if not semantic_success:
                continue

            semantic_result = self.semantic_module.transmit(
                image=self.semantic_images[ue.uid - 1],
                sinr_db=float(10.0 * np.log10(max(mean_sinr, 1e-10))),
                compression_ratio=compression_ratio,
                task_weights={
                    "quality": 1.0,
                    "cost": self.config.semantic_cost_weight,
                },
            )
            successful_ues.append(ue)
            successful_results.append((ue, semantic_result.semantic_quality, semantic_result.semantic_utility))
            semantic_qualities.append(float(semantic_result.semantic_quality))
            semantic_utilities.append(float(semantic_result.semantic_utility))
            semantic_costs.append(float(semantic_result.transmission_cost))

        bit_success_count = len(bit_successful_ues)
        success_count = len(successful_ues)
        invalid_count = max(len(selected_ues) - success_count, 0)
        any_success = success_count > 0
        any_bit_success = bit_success_count > 0
        mean_rate = float(np.mean(selected_rates)) if selected_rates else 0.0
        mean_selected_sinr = float(np.mean(selected_sinrs)) if selected_sinrs else 0.0
        semantic_quality = float(np.mean(semantic_qualities)) if semantic_qualities else 0.0
        semantic_utility = float(np.mean(semantic_utilities)) if semantic_utilities else 0.0
        semantic_cost = float(np.mean(semantic_costs)) if semantic_costs else 0.0

        self._update_aoi_multi(bit_successful_ues)
        self._update_saoi_multi(successful_results)
        energy_info = self._update_energy(
            selected_ues[0] if selected_ues else None,
            charge_flag,
            tx_user_count=len(selected_ues),
        )

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

        if success_count > 0:
            reward += success_count * self.config.success_reward + float(np.sum(semantic_utilities))
        if invalid_count > 0:
            reward -= invalid_count * self.config.invalid_service_penalty

        if self.uav.energy_state in {"return", "charging", "resume"}:
            reward -= self.config.charge_step_penalty

        if self.uav.energy_state == "depleted":
            reward -= self.config.depleted_penalty

        self.current_step += 1
        done = self.current_step >= self.config.max_steps or (
            self.config.terminate_on_depleted and self.uav.energy_state == "depleted"
        )

        selected_ue = selected_ues[0] if selected_ues else None
        info = {
            "mean_saoi": mean_saoi,
            "saoi": self.saoi.copy(),
            "mean_aoi": float(np.mean(self.aoi)),
            "aoi": self.aoi.copy(),
            "semantic_scores": self.semantic_scores.copy(),
            "semantic_quality": float(semantic_quality),
            "semantic_utility": float(semantic_utility),
            "semantic_cost": float(semantic_cost),
            "semantic_attempt": semantic_attempt_count > 0,
            "semantic_attempt_count": int(semantic_attempt_count),
            "semantic_compression_ratio": compression_ratio,
            "selected_ue": None if selected_ue is None else selected_ue.uid,
            "selected_ues": [ue.uid for ue in selected_ues],
            "selected_target_idx": int(target_indices[0]) if target_indices else -1,
            "selected_target_indices": list(target_indices),
            "selected_user_count": int(len(selected_ues)),
            "success": any_success,
            "success_count": int(success_count),
            "bit_success": any_bit_success,
            "bit_success_count": int(bit_success_count),
            "bit_required_rate_mbps": float(bit_required_rate / 1e6),
            "semantic_required_rate_mbps": float(semantic_required_rate / 1e6),
            "rate": mean_rate,
            "per_user_rates": {ue.uid: float(per_user.get(ue.uid, {}).get("rate", 0.0)) for ue in selected_ues},
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "queue_norm": float(queue_norm),
            "drift": float(drift),
            "saoi_improvement": float(saoi_improvement),
            "scheduler_mode": self.config.scheduler_mode,
            "assigned_rbs": link_info["assigned_rbs"],
            "assigned_rbs_map": link_info.get("assigned_rbs_map", {}),
            "mean_sinr": mean_selected_sinr if selected_sinrs else float(link_info["mean_sinr"]),
            "covered_ues": int(link_info["covered_ues"]),
            "bandwidth_alloc": link_info["bandwidth_alloc"],
            "power_alloc": link_info["power_alloc"],
            "mean_bandwidth_alloc": float(link_info["mean_bandwidth_alloc"]),
            "mean_power_alloc": float(link_info["mean_power_alloc"]),
            "sum_bandwidth_alloc": float(link_info["sum_bandwidth_alloc"]),
            "sum_power_alloc": float(link_info["sum_power_alloc"]),
            "resource_allocation_mode": link_info["resource_allocation_mode"],
            "action_ax": float(np.clip(action[0], -1.0, 1.0)),
            "action_ay": float(np.clip(action[1], -1.0, 1.0)),
            "action_charge_bias": float(np.clip(action[2], 0.0, 1.0)),
            "action_service_target": float(np.clip(action[3], 0.0, 1.0)) if not self.config.multi_user_association else 0.0,
            "action_association_scores": np.clip(action[3:-1], 0.0, 1.0).copy() if self.config.multi_user_association else np.array([], dtype=float),
            "action_semantic_ratio": float(np.clip(action[-1], 0.0, 1.0)),
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info
