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
