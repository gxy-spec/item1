from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from env.channel import OFDMAA2GChannel, OFDMAScheduler

from .aoi_energy_env import AoIEnvConfig, SingleUAVAoIEnv


@dataclass
class OFDMAAoIEnvConfig(AoIEnvConfig):
    total_bandwidth: float = 10e6
    num_rbs: int = 16
    total_transmit_power: float = 0.1
    noise_power_density: float = 1e-20
    scheduler_mode: str = "aoi_weighted"


class SingleUAVOFDMAAoIEnv(SingleUAVAoIEnv):
    """单 UAV、多用户 OFDMA、AoI 优化环境。"""

    def __init__(self, config: OFDMAAoIEnvConfig | None = None):
        self.config = config or OFDMAAoIEnvConfig()
        super().__init__(self.config)
        self.ofdma_channel = OFDMAA2GChannel(
            a=9.61,
            b=0.16,
            eta_los=1.6,
            eta_nlos=23.0,
            fc=2.4e9,
            c=3e8,
            total_bandwidth=self.config.total_bandwidth,
            num_rbs=self.config.num_rbs,
            total_transmit_power=self.config.total_transmit_power,
            noise_power_density=self.config.noise_power_density,
        )

    def step(self, action: int) -> Tuple:
        charge_flag, movement_idx, service_idx = self.decode_action(int(action))
        forced_return = False
        if self.config.energy_hard_constraint:
            charge_flag, movement_idx, service_idx, forced_return = self._enforce_energy_hard_constraint(
                charge_flag,
                movement_idx,
                service_idx,
            )
        prev_queue = self.virtual_energy_queue
        prev_mean_aoi = float(self.aoi.mean())

        self._apply_action(charge_flag, movement_idx)
        self._advance_entities()

        selected_ue = None if service_idx == 0 else self.ues[service_idx - 1]
        link_info = self._get_ofdma_link_info(selected_ue)
        rate = float(link_info["rate"])
        success = False
        if selected_ue is not None and self.uav.energy_state == "normal":
            required_rate = self.config.packet_size_bits / self.config.delta_t
            success = rate >= required_rate

        self._update_aoi(selected_ue, success)
        energy_info = self._update_energy(selected_ue, charge_flag)

        mean_aoi = float(self.aoi.mean())
        self.virtual_energy_queue = self._update_virtual_energy_queue(prev_queue, energy_info)
        drift = 0.5 * (self.virtual_energy_queue ** 2 - prev_queue ** 2) / (self.energy_model.E_max ** 2)
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
            "success": success,
            "rate": rate,
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "queue_norm": float(queue_norm),
            "drift": float(drift),
            "aoi_improvement": float(aoi_improvement),
            "forced_return": forced_return,
            "scheduler_mode": self.config.scheduler_mode,
            "assigned_rbs": link_info["assigned_rbs"],
            "mean_sinr": float(link_info["mean_sinr"]),
            "covered_ues": int(link_info["covered_ues"]),
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info

    def _build_scheduler_assignment(self) -> Dict[int, List[int]]:
        if self.config.scheduler_mode == "round_robin":
            return OFDMAScheduler.round_robin(self.ofdma_channel, self.uav, self.ues)
        if self.config.scheduler_mode == "equal_share":
            return OFDMAScheduler.equal_share(self.ofdma_channel, self.uav, self.ues)
        if self.config.scheduler_mode == "aoi_weighted":
            aoi_map = {ue.uid: float(self.aoi[ue.uid - 1]) for ue in self.ues}
            return OFDMAScheduler.aoi_weighted(self.ofdma_channel, self.uav, self.ues, aoi_map)
        raise ValueError(f"Unsupported scheduler_mode: {self.config.scheduler_mode}")

    def _get_ofdma_link_info(self, selected_ue) -> Dict:
        if selected_ue is None or self.uav.energy_state != "normal":
            return {
                "rate": 0.0,
                "assigned_rbs": [],
                "mean_sinr": 0.0,
                "covered_ues": len(self.ofdma_channel.find_covered_ues(self.uav, self.ues)),
            }

        assignment = self._build_scheduler_assignment()
        metrics = self.ofdma_channel.compute_user_rate(
            serving_uav=self.uav,
            ue=selected_ue,
            serving_assignment=assignment,
            interfering_allocations=None,
        )
        return {
            "rate": float(metrics["rate"]),
            "assigned_rbs": list(metrics["assigned_rbs"]),
            "mean_sinr": float(metrics["mean_sinr"]),
            "covered_ues": len(self.ofdma_channel.find_covered_ues(self.uav, self.ues)),
        }
