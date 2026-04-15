import math
from typing import List, Tuple

import numpy as np


class A2GChannel:
    """Air-to-ground信道模型，支持覆盖范围内多用户SINR计算。"""

    def __init__(
        self,
        a: float = 9.61,
        b: float = 0.16,
        eta_los: float = 1.6,
        eta_nlos: float = 23.0,
        fc: float = 2.4e9,
        c: float = 3e8,
        bandwidth: float = 10e6,
        transmit_power: float = 0.1,
        noise_power: float = 1e-9,
    ):
        """初始化A2G通道模型。

        Args:
            a: LoS概率参数 a。
            b: LoS概率参数 b。
            eta_los: LoS附加损耗 (dB)。
            eta_nlos: NLoS附加损耗 (dB)。
            fc: 载波频率 (Hz)。
            c: 光速 (m/s)。
            bandwidth: 带宽 (Hz)。
            transmit_power: UE发射功率 (W)。
            noise_power: 噪声功率 (W)。
        """
        self.a = a
        self.b = b
        self.eta_los = eta_los
        self.eta_nlos = eta_nlos
        self.fc = fc
        self.c = c
        self.bandwidth = bandwidth
        self.transmit_power = transmit_power
        self.noise_power = noise_power

    def distance(self, uav, ue) -> float:
        """计算UAV与UE之间的三维距离。"""
        dx = uav.xy[0] - ue.xy[0]
        dy = uav.xy[1] - ue.xy[1]
        return math.sqrt(dx * dx + dy * dy + uav.height * uav.height)

    def elevation_angle(self, uav, ue) -> float:
        """计算仰角 (度)。"""
        d = self.distance(uav, ue)
        ratio = min(max(uav.height / d, 0.0), 1.0)
        return math.degrees(math.asin(ratio))

    def los_probability(self, theta_deg: float) -> float:
        """计算LoS概率。"""
        return 1.0 / (1.0 + self.a * math.exp(-self.b * (theta_deg - self.a)))

    def fspl(self, distance: float) -> float:
        """自由空间路径损耗 (dB)。"""
        if distance <= 0:
            return 0.0
        return 20.0 * math.log10(4.0 * math.pi * self.fc * distance / self.c)

    def path_loss(self, uav, ue) -> float:
        """计算A2G路径损耗 (dB)。"""
        d = self.distance(uav, ue)
        theta = self.elevation_angle(uav, ue)
        p_los = self.los_probability(theta)
        fspl = self.fspl(d)
        return fspl + p_los * self.eta_los + (1.0 - p_los) * self.eta_nlos

    def power_gain(self, uav, ue) -> float:
        """计算链路增益幅度。"""
        pl = self.path_loss(uav, ue)
        return 10.0 ** (-pl / 20.0)

    def find_covered_ues(self, uav, ue_list: List) -> List:
        """返回在UAV服务范围内的UE列表。"""
        covered = []
        for ue in ue_list:
            dx = uav.xy[0] - ue.xy[0]
            dy = uav.xy[1] - ue.xy[1]
            planar_distance = math.hypot(dx, dy)
            if planar_distance <= uav.service_radius:
                covered.append(ue)
        return covered

    def compute_sinr_and_rate(self, uav, ue_list: List, interfering_uavs: List = None) -> List[dict]:
        """计算UAV覆盖区内每个UE的SINR和速率。

        这里将其他UAV视为共信道干扰源，与热力图中的SINR定义保持一致。
        """
        covered = self.find_covered_ues(uav, ue_list)
        if len(covered) == 0:
            return []

        interfering_uavs = [] if interfering_uavs is None else interfering_uavs
        gains = [self.power_gain(uav, ue) for ue in covered]
        power = self.transmit_power
        results = []

        for idx, ue in enumerate(covered):
            signal = power * gains[idx] ** 2
            interference = 0.0
            for other_uav in interfering_uavs:
                if other_uav is uav:
                    continue
                interference_gain = self.power_gain(other_uav, ue)
                interference += power * interference_gain ** 2
            sinr = signal / (interference + self.noise_power)
            rate = self.bandwidth * math.log2(1.0 + sinr)
            results.append(
                {
                    "ue": ue,
                    "distance": self.distance(uav, ue),
                    "theta_deg": self.elevation_angle(uav, ue),
                    "p_los": self.los_probability(self.elevation_angle(uav, ue)),
                    "path_loss_db": self.path_loss(uav, ue),
                    "gain": gains[idx],
                    "sinr": sinr,
                    "rate": rate,
                }
            )

        return results

    def compute_heatmap(
        self,
        uav_list: List,
        ue_list: List,
        area_bounds: Tuple[float, float, float, float],
        grid_size: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算包含多UAV干扰的SINR热力图（矢量化版本）。

        该热力图计算整个区域内每个点的SINR，考虑所有UAV的信号和干扰。
        """
        xmin, xmax, ymin, ymax = area_bounds
        x = np.linspace(xmin, xmax, grid_size)
        y = np.linspace(ymin, ymax, grid_size)
        xx, yy = np.meshgrid(x, y)

        # 初始化SINR地图
        sinr_map = np.full_like(xx, np.nan)
        coverage_mask = np.zeros_like(xx, dtype=bool)

        # 对每个UAV计算信号和干扰贡献（矢量化）
        for idx, uav in enumerate(uav_list):
            # 计算到该UAV的3D距离
            dz = uav.height
            dist_3d = np.sqrt((xx - uav.xy[0]) ** 2 + (yy - uav.xy[1]) ** 2 + dz ** 2)

            # 避免除以零
            dist_3d = np.maximum(dist_3d, 1.0)

            # 计算仰角和LoS概率
            theta = np.degrees(np.arcsin(np.clip(dz / dist_3d, 0.0, 1.0)))
            p_los = 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))

            # 计算路径损失
            fspl = 20.0 * np.log10(4.0 * math.pi * self.fc * dist_3d / self.c)
            pl = fspl + p_los * self.eta_los + (1.0 - p_los) * self.eta_nlos
            gain = 10.0 ** (-pl / 20.0)

            # 只在服务范围内计算
            planar_dist = np.sqrt((xx - uav.xy[0]) ** 2 + (yy - uav.xy[1]) ** 2)
            service_mask = planar_dist <= uav.service_radius
            coverage_mask |= service_mask

            # 计算该UAV的信号功率
            signal_power = self.transmit_power * gain ** 2

            # 计算其他UAV的总干扰
            interference_power = np.zeros_like(xx)
            for other_uav in uav_list:
                if other_uav is not uav:
                    dz_int = other_uav.height
                    dist_3d_int = np.sqrt((xx - other_uav.xy[0]) ** 2 + (yy - other_uav.xy[1]) ** 2 + dz_int ** 2)
                    dist_3d_int = np.maximum(dist_3d_int, 1.0)

                    theta_int = np.degrees(np.arcsin(np.clip(dz_int / dist_3d_int, 0.0, 1.0)))
                    p_los_int = 1.0 / (1.0 + self.a * np.exp(-self.b * (theta_int - self.a)))

                    fspl_int = 20.0 * np.log10(4.0 * math.pi * self.fc * dist_3d_int / self.c)
                    pl_int = fspl_int + p_los_int * self.eta_los + (1.0 - p_los_int) * self.eta_nlos
                    gain_int = 10.0 ** (-pl_int / 20.0)

                    interference_power += self.transmit_power * gain_int ** 2

            # 计算SINR
            sinr = signal_power / (interference_power + self.noise_power)

            # 在该UAV覆盖范围内，保存最大SINR
            if idx == 0:
                sinr_map[service_mask] = sinr[service_mask]
            else:
                better_mask = service_mask & (sinr > sinr_map)
                sinr_map[better_mask] = sinr[better_mask]
                new_coverage = service_mask & np.isnan(sinr_map)
                sinr_map[new_coverage] = sinr[new_coverage]

        return xx, yy, sinr_map
