from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ResourceAllocationResult:
    bandwidth: np.ndarray
    power: np.ndarray


class UniformResourceAllocator:
    def __init__(self, total_bandwidth: float, total_power: float, eps: float = 1e-12):
        self.total_bandwidth = float(total_bandwidth)
        self.total_power = float(total_power)
        self.eps = float(eps)

    def allocate(self, active_user_indices: Sequence[int], num_users: int) -> ResourceAllocationResult:
        bandwidth = np.zeros(num_users, dtype=float)
        power = np.zeros(num_users, dtype=float)

        unique_active = sorted({int(idx) for idx in active_user_indices if 0 <= int(idx) < num_users})
        if not unique_active:
            return ResourceAllocationResult(bandwidth=bandwidth, power=power)

        active_count = len(unique_active)
        per_user_bandwidth = self.total_bandwidth / max(active_count, 1)
        per_user_power = self.total_power / max(active_count, 1)

        for idx in unique_active:
            bandwidth[idx] = per_user_bandwidth
            power[idx] = per_user_power

        bandwidth_sum = float(np.sum(bandwidth))
        power_sum = float(np.sum(power))
        if bandwidth_sum > self.total_bandwidth + self.eps:
            bandwidth *= self.total_bandwidth / max(bandwidth_sum, self.eps)
        if power_sum > self.total_power + self.eps:
            power *= self.total_power / max(power_sum, self.eps)

        return ResourceAllocationResult(bandwidth=bandwidth, power=power)


class KKTResourceAllocator:
    def __init__(
        self,
        total_bandwidth: float,
        total_power: float,
        noise: float,
        eps: float = 1e-12,
        max_iter: int = 80,
        power_floor_ratio: float = 0.40,
        bandwidth_floor_ratio: float = 0.40,
        weight_exponent: float = 0.5,
        min_weight_ratio: float = 0.5,
        max_weight_ratio: float = 2.0,
    ):
        self.total_bandwidth = float(total_bandwidth)
        self.total_power = float(total_power)
        self.noise = float(noise)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.power_floor_ratio = float(np.clip(power_floor_ratio, 0.0, 0.95))
        self.bandwidth_floor_ratio = float(np.clip(bandwidth_floor_ratio, 0.0, 0.95))
        self.weight_exponent = float(np.clip(weight_exponent, self.eps, 2.0))
        self.min_weight_ratio = float(max(min_weight_ratio, self.eps))
        self.max_weight_ratio = float(max(max_weight_ratio, self.min_weight_ratio))
        self.uniform_fallback = UniformResourceAllocator(total_bandwidth=total_bandwidth, total_power=total_power, eps=eps)

    def allocate(
        self,
        active_user_indices: Sequence[int],
        num_users: int,
        channel_gains: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> ResourceAllocationResult:
        bandwidth = np.zeros(num_users, dtype=float)
        power = np.zeros(num_users, dtype=float)
        active = self._sanitize_active(active_user_indices, num_users)
        if not active:
            return ResourceAllocationResult(bandwidth=bandwidth, power=power)

        gains, active_weights = self._prepare_inputs(active, channel_gains, weights)
        active_power = self._allocate_power(active, gains, active_weights)
        if active_power is None:
            return self.uniform_fallback.allocate(active, num_users)

        active_bandwidth = self._allocate_bandwidth(gains, active_power, active_weights)
        bandwidth[active] = active_bandwidth
        power[active] = active_power
        return self._clip_totals(ResourceAllocationResult(bandwidth=bandwidth, power=power))

    def _sanitize_active(self, active_user_indices: Sequence[int], num_users: int) -> list[int]:
        return sorted({int(idx) for idx in active_user_indices if 0 <= int(idx) < num_users})

    def _prepare_inputs(
        self,
        active: Sequence[int],
        channel_gains: Sequence[float],
        weights: Sequence[float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        gains = np.asarray(channel_gains, dtype=float).reshape(-1)
        if gains.size <= max(active):
            raise ValueError("channel_gains must provide one gain per user")
        active_gains = np.maximum(gains[np.asarray(active, dtype=int)], self.eps)

        if weights is None:
            active_weights = np.ones(len(active), dtype=float)
        else:
            weight_array = np.asarray(weights, dtype=float).reshape(-1)
            if weight_array.size <= max(active):
                raise ValueError("weights must provide one weight per user")
            active_weights = np.maximum(weight_array[np.asarray(active, dtype=int)], self.eps)
        smoothed_weights = self._smooth_weights(active_weights)
        return active_gains, smoothed_weights

    def _smooth_weights(self, weights: np.ndarray) -> np.ndarray:
        mean_weight = max(float(np.mean(weights)), self.eps)
        normalized = np.maximum(weights / mean_weight, self.eps)
        smoothed = np.power(normalized, self.weight_exponent)
        clipped = np.clip(smoothed, self.min_weight_ratio, self.max_weight_ratio)
        return clipped

    def _allocate_power(self, active: Sequence[int], gains: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        active_count = len(active)
        if active_count == 0:
            return np.zeros(0, dtype=float)
        if len(active) == 1:
            return np.array([self.total_power], dtype=float)

        base_power = self.total_power * self.power_floor_ratio / active_count
        remaining_power_budget = max(self.total_power - base_power * active_count, 0.0)
        if remaining_power_budget <= self.eps:
            return np.full(active_count, self.total_power / active_count, dtype=float)

        noise = max(self.noise, self.eps)
        low = self.eps
        high = max(np.max(weights / noise), self.eps)
        for _ in range(self.max_iter):
            high *= 2.0
            candidate = np.maximum(weights / high - noise / gains, 0.0)
            if float(np.sum(candidate)) <= remaining_power_budget + self.eps:
                break
        else:
            return None

        for _ in range(self.max_iter):
            mid = 0.5 * (low + high)
            candidate = np.maximum(weights / mid - noise / gains, 0.0)
            if float(np.sum(candidate)) > remaining_power_budget:
                low = mid
            else:
                high = mid

        power = np.maximum(weights / high - noise / gains, 0.0)
        total = float(np.sum(power))
        if not np.isfinite(total) or total <= self.eps:
            return None
        power *= remaining_power_budget / total
        power += base_power
        return power

    def _allocate_bandwidth(self, gains: np.ndarray, power: np.ndarray, weights: np.ndarray) -> np.ndarray:
        active_count = power.size
        if active_count == 0:
            return np.zeros(0, dtype=float)
        base_bandwidth = self.total_bandwidth * self.bandwidth_floor_ratio / active_count
        remaining_bandwidth_budget = max(self.total_bandwidth - base_bandwidth * active_count, 0.0)
        if remaining_bandwidth_budget <= self.eps:
            return np.full(active_count, self.total_bandwidth / active_count, dtype=float)

        noise = max(self.noise, self.eps)
        sinr = np.maximum(power * gains / noise, 0.0)
        bw_weights = weights * np.log2(1.0 + sinr)
        total_bw_weight = float(np.sum(bw_weights))
        if total_bw_weight <= self.eps:
            return np.full_like(power, self.total_bandwidth / max(power.size, 1), dtype=float)
        bandwidth = remaining_bandwidth_budget * bw_weights / total_bw_weight
        bandwidth += base_bandwidth
        return bandwidth

    def _clip_totals(self, result: ResourceAllocationResult) -> ResourceAllocationResult:
        bandwidth = np.asarray(result.bandwidth, dtype=float)
        power = np.asarray(result.power, dtype=float)
        bandwidth_sum = float(np.sum(bandwidth))
        power_sum = float(np.sum(power))
        if bandwidth_sum > self.total_bandwidth + self.eps:
            bandwidth *= self.total_bandwidth / max(bandwidth_sum, self.eps)
        if power_sum > self.total_power + self.eps:
            power *= self.total_power / max(power_sum, self.eps)
        return ResourceAllocationResult(bandwidth=bandwidth, power=power)
