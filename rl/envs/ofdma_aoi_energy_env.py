from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from env.channel import OFDMAA2GChannel, OFDMAScheduler
from rl.resources import KKTResourceAllocator, ResourceAllocationResult, UniformResourceAllocator

from .aoi_energy_env import AoIEnvConfig, SingleUAVAoIEnv


@dataclass
class OFDMAAoIEnvConfig(AoIEnvConfig):
    total_bandwidth: float = 10e6
    num_rbs: int = 16
    total_transmit_power: float = 0.1
    noise_power_density: float = 1e-20
    scheduler_mode: str = "aoi_weighted"
    enable_resource_allocation: bool = True
    resource_allocation_mode: str = "fixed"


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
        self.uniform_allocator = UniformResourceAllocator(
            total_bandwidth=self.config.total_bandwidth,
            total_power=self.config.total_transmit_power,
        )
        self.kkt_allocator = KKTResourceAllocator(
            total_bandwidth=self.config.total_bandwidth,
            total_power=self.config.total_transmit_power,
            noise=self.config.noise_power_density * self.config.total_bandwidth,
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
            "bandwidth_alloc": link_info["bandwidth_alloc"],
            "power_alloc": link_info["power_alloc"],
            "mean_bandwidth_alloc": float(link_info["mean_bandwidth_alloc"]),
            "mean_power_alloc": float(link_info["mean_power_alloc"]),
            "sum_bandwidth_alloc": float(link_info["sum_bandwidth_alloc"]),
            "sum_power_alloc": float(link_info["sum_power_alloc"]),
            "resource_allocation_mode": link_info["resource_allocation_mode"],
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info

    def _build_scheduler_assignment(self, candidate_ues: Sequence | None = None) -> Dict[int, List[int]]:
        ue_list = list(candidate_ues) if candidate_ues is not None else self.ues
        if self.config.scheduler_mode == "round_robin":
            return OFDMAScheduler.round_robin(self.ofdma_channel, self.uav, ue_list)
        if self.config.scheduler_mode == "equal_share":
            return OFDMAScheduler.equal_share(self.ofdma_channel, self.uav, ue_list)
        if self.config.scheduler_mode == "aoi_weighted":
            aoi_map = {ue.uid: float(self.aoi[ue.uid - 1]) for ue in ue_list}
            return OFDMAScheduler.aoi_weighted(self.ofdma_channel, self.uav, ue_list, aoi_map)
        raise ValueError(f"Unsupported scheduler_mode: {self.config.scheduler_mode}")

    def _empty_link_info(self) -> Dict:
        empty_alloc = np.zeros(self.config.num_ues, dtype=float)
        return {
            "rate": 0.0,
            "assigned_rbs": [],
            "mean_sinr": 0.0,
            "covered_ues": 0,
            "bandwidth_alloc": empty_alloc.copy(),
            "power_alloc": empty_alloc.copy(),
            "mean_bandwidth_alloc": 0.0,
            "mean_power_alloc": 0.0,
            "sum_bandwidth_alloc": 0.0,
            "sum_power_alloc": 0.0,
            "resource_allocation_mode": self.config.resource_allocation_mode,
        }

    def _resolve_active_ues(self, active_user_indices: Sequence[int] | None = None) -> List:
        if active_user_indices is None:
            return []

        active_ues = []
        for user_idx in active_user_indices:
            if 0 <= int(user_idx) < self.config.num_ues:
                ue = self.ues[int(user_idx)]
                if self._is_covered(ue):
                    active_ues.append(ue)
        return active_ues

    def _resource_allocation_channel_gains(self, active_ues: Sequence) -> np.ndarray:
        gains = np.zeros(self.config.num_ues, dtype=float)
        for ue in active_ues:
            raw_gain = float(self.ofdma_channel.power_gain(self.uav, ue))
            gains[ue.uid - 1] = raw_gain**2
        return gains

    def _resource_allocation_weights(self, active_ues: Sequence) -> np.ndarray:
        # TODO: replace with SAoI-aware weights in semantic SAoI environments.
        weights = np.zeros(self.config.num_ues, dtype=float)
        for ue in active_ues:
            weights[ue.uid - 1] = 1.0
        return weights

    def _get_ofdma_link_info(self, selected_ue, active_user_indices: Sequence[int] | None = None) -> Dict:
        if selected_ue is None or self.uav.energy_state != "normal":
            return self._empty_link_info()

        resolved_indices = (
            list(active_user_indices)
            if active_user_indices is not None
            else [selected_ue.uid - 1]
        )
        active_ues = self._resolve_active_ues(resolved_indices)
        if selected_ue not in active_ues:
            return self._empty_link_info()

        if self.config.enable_resource_allocation and self.config.resource_allocation_mode == "uniform":
            return self._get_uniform_resource_link_info(selected_ue, active_ues)
        if self.config.enable_resource_allocation and self.config.resource_allocation_mode == "kkt":
            return self._get_kkt_resource_link_info(selected_ue, active_ues)

        return self._get_fixed_resource_link_info(selected_ue, active_ues)

    def _get_ofdma_multiuser_link_info(self, selected_ues: Sequence) -> Dict:
        if self.uav.energy_state != "normal":
            empty = self._empty_link_info()
            empty.update({"per_user": {}, "selected_user_count": 0, "assigned_rbs_map": {}})
            return empty

        active_ues = [ue for ue in selected_ues if self._is_covered(ue)]
        if not active_ues:
            empty = self._empty_link_info()
            empty.update({"per_user": {}, "selected_user_count": 0, "assigned_rbs_map": {}})
            return empty

        if self.config.enable_resource_allocation and self.config.resource_allocation_mode == "uniform":
            return self._get_uniform_multiuser_link_info(active_ues)
        if self.config.enable_resource_allocation and self.config.resource_allocation_mode == "kkt":
            return self._get_kkt_multiuser_link_info(active_ues)

        return self._get_fixed_multiuser_link_info(active_ues)

    def _empty_resource_summary(self) -> ResourceAllocationResult:
        zeros = np.zeros(self.config.num_ues, dtype=float)
        return ResourceAllocationResult(bandwidth=zeros.copy(), power=zeros.copy())

    def _summarize_resource_allocation(self, allocation: ResourceAllocationResult) -> Dict:
        bandwidth = np.asarray(allocation.bandwidth, dtype=float)
        power = np.asarray(allocation.power, dtype=float)
        active_bw = bandwidth[bandwidth > 0.0]
        active_power = power[power > 0.0]
        return {
            "bandwidth_alloc": bandwidth.copy(),
            "power_alloc": power.copy(),
            "mean_bandwidth_alloc": float(np.mean(active_bw)) if active_bw.size else 0.0,
            "mean_power_alloc": float(np.mean(active_power)) if active_power.size else 0.0,
            "sum_bandwidth_alloc": float(np.sum(bandwidth)),
            "sum_power_alloc": float(np.sum(power)),
        }

    def _get_fixed_resource_link_info(self, selected_ue, active_ues: Sequence) -> Dict:
        assignment = self._build_scheduler_assignment(active_ues)
        metrics = self.ofdma_channel.compute_user_rate(
            serving_uav=self.uav,
            ue=selected_ue,
            serving_assignment=assignment,
            interfering_allocations=None,
        )
        rb_power = self.ofdma_channel._rb_power(assignment)
        bandwidth = np.zeros(self.config.num_ues, dtype=float)
        power = np.zeros(self.config.num_ues, dtype=float)
        for ue_uid, rb_list in assignment.items():
            idx = ue_uid - 1
            bandwidth[idx] = len(rb_list) * self.ofdma_channel.rb_bandwidth
            power[idx] = len(rb_list) * rb_power
        allocation = ResourceAllocationResult(bandwidth=bandwidth, power=power)
        summary = self._summarize_resource_allocation(allocation)
        return {
            "rate": float(metrics["rate"]),
            "assigned_rbs": list(metrics["assigned_rbs"]),
            "mean_sinr": float(metrics["mean_sinr"]),
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "fixed",
            **summary,
        }

    def _get_uniform_resource_link_info(self, selected_ue, active_ues: Sequence) -> Dict:
        active_user_indices = [ue.uid - 1 for ue in active_ues]
        allocation = self.uniform_allocator.allocate(
            active_user_indices=active_user_indices,
            num_users=self.config.num_ues,
        )
        user_idx = selected_ue.uid - 1
        user_bandwidth = float(allocation.bandwidth[user_idx])
        user_power = float(allocation.power[user_idx])

        if user_bandwidth <= 0.0 or user_power <= 0.0:
            summary = self._summarize_resource_allocation(allocation)
            return {
                "rate": 0.0,
                "assigned_rbs": [],
                "mean_sinr": 0.0,
                "covered_ues": len(active_ues),
                "resource_allocation_mode": "uniform",
                **summary,
            }

        gain = self.ofdma_channel.power_gain(self.uav, selected_ue)
        noise_power = self.ofdma_channel.noise_power_density * user_bandwidth
        mean_sinr = (user_power * gain**2) / max(noise_power, 1e-12)
        rate = user_bandwidth * math.log2(1.0 + mean_sinr)
        summary = self._summarize_resource_allocation(allocation)
        return {
            "rate": float(rate),
            "assigned_rbs": [],
            "mean_sinr": float(mean_sinr),
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "uniform",
            **summary,
        }

    def _get_kkt_resource_link_info(self, selected_ue, active_ues: Sequence) -> Dict:
        active_user_indices = [ue.uid - 1 for ue in active_ues]
        channel_gains = self._resource_allocation_channel_gains(active_ues)
        weights = self._resource_allocation_weights(active_ues)
        allocation = self.kkt_allocator.allocate(
            active_user_indices=active_user_indices,
            num_users=self.config.num_ues,
            channel_gains=channel_gains,
            weights=weights,
        )
        user_idx = selected_ue.uid - 1
        user_bandwidth = float(allocation.bandwidth[user_idx])
        user_power = float(allocation.power[user_idx])

        if user_bandwidth <= 0.0 or user_power <= 0.0:
            summary = self._summarize_resource_allocation(allocation)
            return {
                "rate": 0.0,
                "assigned_rbs": [],
                "mean_sinr": 0.0,
                "covered_ues": len(active_ues),
                "resource_allocation_mode": "kkt",
                **summary,
            }

        gain = channel_gains[user_idx]
        noise_power = self.ofdma_channel.noise_power_density * user_bandwidth
        mean_sinr = (user_power * gain) / max(noise_power, 1e-12)
        rate = user_bandwidth * math.log2(1.0 + mean_sinr)
        summary = self._summarize_resource_allocation(allocation)
        return {
            "rate": float(rate),
            "assigned_rbs": [],
            "mean_sinr": float(mean_sinr),
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "kkt",
            **summary,
        }

    def _get_fixed_multiuser_link_info(self, active_ues: Sequence) -> Dict:
        assignment = self._build_scheduler_assignment(active_ues)
        rb_power = self.ofdma_channel._rb_power(assignment)
        bandwidth = np.zeros(self.config.num_ues, dtype=float)
        power = np.zeros(self.config.num_ues, dtype=float)
        per_user = {}
        assigned_rbs_map = {}

        for ue in active_ues:
            metrics = self.ofdma_channel.compute_user_rate(
                serving_uav=self.uav,
                ue=ue,
                serving_assignment=assignment,
                interfering_allocations=None,
            )
            idx = ue.uid - 1
            rb_list = list(metrics["assigned_rbs"])
            bandwidth[idx] = len(rb_list) * self.ofdma_channel.rb_bandwidth
            power[idx] = len(rb_list) * rb_power
            assigned_rbs_map[ue.uid] = rb_list
            per_user[ue.uid] = {
                "rate": float(metrics["rate"]),
                "mean_sinr": float(metrics["mean_sinr"]),
                "assigned_rbs": rb_list,
                "bandwidth": float(bandwidth[idx]),
                "power": float(power[idx]),
            }

        allocation = ResourceAllocationResult(bandwidth=bandwidth, power=power)
        summary = self._summarize_resource_allocation(allocation)
        rates = [item["rate"] for item in per_user.values()]
        mean_sinrs = [item["mean_sinr"] for item in per_user.values()]
        return {
            "rate": float(np.mean(rates)) if rates else 0.0,
            "assigned_rbs": [],
            "assigned_rbs_map": assigned_rbs_map,
            "mean_sinr": float(np.mean(mean_sinrs)) if mean_sinrs else 0.0,
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "fixed",
            "per_user": per_user,
            "selected_user_count": len(active_ues),
            **summary,
        }

    def _get_uniform_multiuser_link_info(self, active_ues: Sequence) -> Dict:
        active_user_indices = [ue.uid - 1 for ue in active_ues]
        allocation = self.uniform_allocator.allocate(
            active_user_indices=active_user_indices,
            num_users=self.config.num_ues,
        )
        per_user = {}
        assigned_rbs_map = {}
        rates = []
        mean_sinrs = []

        for ue in active_ues:
            user_idx = ue.uid - 1
            user_bandwidth = float(allocation.bandwidth[user_idx])
            user_power = float(allocation.power[user_idx])
            if user_bandwidth <= 0.0 or user_power <= 0.0:
                mean_sinr = 0.0
                rate = 0.0
            else:
                gain = self.ofdma_channel.power_gain(self.uav, ue)
                noise_power = self.ofdma_channel.noise_power_density * user_bandwidth
                mean_sinr = (user_power * gain**2) / max(noise_power, 1e-12)
                rate = user_bandwidth * math.log2(1.0 + mean_sinr)
            per_user[ue.uid] = {
                "rate": float(rate),
                "mean_sinr": float(mean_sinr),
                "assigned_rbs": [],
                "bandwidth": float(user_bandwidth),
                "power": float(user_power),
            }
            assigned_rbs_map[ue.uid] = []
            rates.append(float(rate))
            mean_sinrs.append(float(mean_sinr))

        summary = self._summarize_resource_allocation(allocation)
        return {
            "rate": float(np.mean(rates)) if rates else 0.0,
            "assigned_rbs": [],
            "assigned_rbs_map": assigned_rbs_map,
            "mean_sinr": float(np.mean(mean_sinrs)) if mean_sinrs else 0.0,
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "uniform",
            "per_user": per_user,
            "selected_user_count": len(active_ues),
            **summary,
        }

    def _get_kkt_multiuser_link_info(self, active_ues: Sequence) -> Dict:
        active_user_indices = [ue.uid - 1 for ue in active_ues]
        channel_gains = self._resource_allocation_channel_gains(active_ues)
        weights = self._resource_allocation_weights(active_ues)
        allocation = self.kkt_allocator.allocate(
            active_user_indices=active_user_indices,
            num_users=self.config.num_ues,
            channel_gains=channel_gains,
            weights=weights,
        )
        per_user = {}
        assigned_rbs_map = {}
        rates = []
        mean_sinrs = []

        for ue in active_ues:
            user_idx = ue.uid - 1
            user_bandwidth = float(allocation.bandwidth[user_idx])
            user_power = float(allocation.power[user_idx])
            if user_bandwidth <= 0.0 or user_power <= 0.0:
                mean_sinr = 0.0
                rate = 0.0
            else:
                gain = channel_gains[user_idx]
                noise_power = self.ofdma_channel.noise_power_density * user_bandwidth
                mean_sinr = (user_power * gain) / max(noise_power, 1e-12)
                rate = user_bandwidth * math.log2(1.0 + mean_sinr)
            per_user[ue.uid] = {
                "rate": float(rate),
                "mean_sinr": float(mean_sinr),
                "assigned_rbs": [],
                "bandwidth": float(user_bandwidth),
                "power": float(user_power),
            }
            assigned_rbs_map[ue.uid] = []
            rates.append(float(rate))
            mean_sinrs.append(float(mean_sinr))

        summary = self._summarize_resource_allocation(allocation)
        return {
            "rate": float(np.mean(rates)) if rates else 0.0,
            "assigned_rbs": [],
            "assigned_rbs_map": assigned_rbs_map,
            "mean_sinr": float(np.mean(mean_sinrs)) if mean_sinrs else 0.0,
            "covered_ues": len(active_ues),
            "resource_allocation_mode": "kkt",
            "per_user": per_user,
            "selected_user_count": len(active_ues),
            **summary,
        }
