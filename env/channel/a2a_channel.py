import math
from typing import Tuple

import numpy as np


class A2AChannel:
    """UAV到HAP的A2A链路物理通道模型。"""

    def __init__(
        self,
        beta0: float = 1e-3,
        kappa: float = 1e-4,
        bandwidth: float = 10e6,
        transmit_power: float = 0.5,
        noise_power: float = 1e-9,
    ):
        """初始化A2A通道模型。

        Args:
            beta0: 基准增益系数。
            kappa: 衰减因子。
            bandwidth: 带宽 (Hz)。
            transmit_power: UAV发射功率 (W)。
            noise_power: 噪声功率 (W)。
        """
        self.beta0 = beta0
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.transmit_power = transmit_power
        self.noise_power = noise_power

    def distance(self, uav, hap) -> float:
        """计算A2A链路距离。"""
        dx = uav.xy[0] - hap.xy[0]
        dy = uav.xy[1] - hap.xy[1]
        dz = uav.height - hap.height
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def gain(self, uav, hap) -> float:
        """计算A2A信道增益。"""
        d = self.distance(uav, hap)
        if d <= 0:
            return 0.0
        return self.beta0 * d ** -2 * math.exp(-self.kappa * d)

    def snr(self, uav, hap) -> float:
        """计算A2A信噪比。"""
        g = self.gain(uav, hap)
        return self.transmit_power * g / self.noise_power

    def rate(self, uav, hap) -> float:
        """计算A2A速率（比特/秒）。"""
        snr_value = self.snr(uav, hap)
        return self.bandwidth * math.log2(1.0 + snr_value)

    def compute_link_metrics(self, uav, hap) -> dict:
        """返回A2A链路的距离、增益、SNR和速率。"""
        d = self.distance(uav, hap)
        g = self.gain(uav, hap)
        snr_value = self.snr(uav, hap)
        rate_value = self.rate(uav, hap)
        return {
            "distance": d,
            "gain": g,
            "snr": snr_value,
            "rate": rate_value,
        }
