import math
from typing import Dict, Iterable, List, Optional, Tuple


class OFDMAA2GChannel:
    """多用户 OFDMA A2G 信道模型。

    这个版本和当前单带宽 Shannon 模型的主要区别是：
    - 总带宽被离散成多个 RB（resource blocks）
    - 同一架 UAV 内部采用 OFDMA，UE 之间正交分配 RB，不互相干扰
    - 只有不同 UAV 在复用同一个 RB 时，才产生共信道干扰
    """

    def __init__(
        self,
        a: float = 9.61,
        b: float = 0.16,
        eta_los: float = 1.6,
        eta_nlos: float = 23.0,
        fc: float = 2.4e9,
        c: float = 3e8,
        total_bandwidth: float = 10e6,
        num_rbs: int = 16,
        total_transmit_power: float = 0.1,
        noise_power_density: float = 1e-20,
    ):
        self.a = a
        self.b = b
        self.eta_los = eta_los
        self.eta_nlos = eta_nlos
        self.fc = fc
        self.c = c
        self.total_bandwidth = total_bandwidth
        self.num_rbs = num_rbs
        self.total_transmit_power = total_transmit_power
        self.noise_power_density = noise_power_density

    @property
    def rb_bandwidth(self) -> float:
        return self.total_bandwidth / self.num_rbs

    @property
    def rb_noise_power(self) -> float:
        return self.noise_power_density * self.rb_bandwidth

    def distance(self, uav, ue) -> float:
        dx = uav.xy[0] - ue.xy[0]
        dy = uav.xy[1] - ue.xy[1]
        return math.sqrt(dx * dx + dy * dy + uav.height * uav.height)

    def elevation_angle(self, uav, ue) -> float:
        d = self.distance(uav, ue)
        ratio = min(max(uav.height / d, 0.0), 1.0)
        return math.degrees(math.asin(ratio))

    def los_probability(self, theta_deg: float) -> float:
        return 1.0 / (1.0 + self.a * math.exp(-self.b * (theta_deg - self.a)))

    def fspl(self, distance: float) -> float:
        if distance <= 0:
            return 0.0
        return 20.0 * math.log10(4.0 * math.pi * self.fc * distance / self.c)

    def path_loss(self, uav, ue) -> float:
        d = self.distance(uav, ue)
        theta = self.elevation_angle(uav, ue)
        p_los = self.los_probability(theta)
        free_space = self.fspl(d)
        return free_space + p_los * self.eta_los + (1.0 - p_los) * self.eta_nlos

    def power_gain(self, uav, ue) -> float:
        pl = self.path_loss(uav, ue)
        return 10.0 ** (-pl / 20.0)

    def find_covered_ues(self, uav, ue_list: List) -> List:
        covered = []
        for ue in ue_list:
            dx = uav.xy[0] - ue.xy[0]
            dy = uav.xy[1] - ue.xy[1]
            planar_distance = math.hypot(dx, dy)
            if planar_distance <= uav.service_radius:
                covered.append(ue)
        return covered

    def build_round_robin_assignment(self, uav, ue_list: List) -> Dict[int, List[int]]:
        """构造一个最简单的 OFDMA RB 分配。

        返回：
        - key: ue.uid
        - value: 分配给该 UE 的 RB 索引列表
        """
        covered = self.find_covered_ues(uav, ue_list)
        if not covered:
            return {}

        assignment = {ue.uid: [] for ue in covered}
        for rb_idx in range(self.num_rbs):
            ue = covered[rb_idx % len(covered)]
            assignment[ue.uid].append(rb_idx)
        return assignment

    def _count_active_rbs(self, assignment: Dict[int, List[int]]) -> int:
        return sum(len(rb_list) for rb_list in assignment.values())

    def _rb_power(self, assignment: Dict[int, List[int]]) -> float:
        active_rbs = max(1, self._count_active_rbs(assignment))
        return self.total_transmit_power / active_rbs

    def _rb_is_used(self, assignment: Dict[int, List[int]], rb_idx: int) -> bool:
        return any(rb_idx in rb_list for rb_list in assignment.values())

    def compute_rb_sinr(
        self,
        serving_uav,
        ue,
        rb_idx: int,
        serving_assignment: Dict[int, List[int]],
        interfering_allocations: Optional[Iterable[Tuple[object, Dict[int, List[int]]]]] = None,
    ) -> float:
        """计算单个 RB 上的 SINR。"""
        interfering_allocations = [] if interfering_allocations is None else interfering_allocations

        serving_gain = self.power_gain(serving_uav, ue)
        signal = self._rb_power(serving_assignment) * serving_gain ** 2

        interference = 0.0
        for other_uav, other_assignment in interfering_allocations:
            if not self._rb_is_used(other_assignment, rb_idx):
                continue
            interference_gain = self.power_gain(other_uav, ue)
            interference += self._rb_power(other_assignment) * interference_gain ** 2

        return signal / (interference + self.rb_noise_power)

    def compute_user_rate(
        self,
        serving_uav,
        ue,
        serving_assignment: Dict[int, List[int]],
        interfering_allocations: Optional[Iterable[Tuple[object, Dict[int, List[int]]]]] = None,
    ) -> dict:
        """计算某个 UE 在其分配 RB 上的总速率。"""
        assigned_rbs = serving_assignment.get(ue.uid, [])
        if not assigned_rbs:
            return {
                "ue": ue,
                "assigned_rbs": [],
                "sinr_per_rb": [],
                "mean_sinr": 0.0,
                "rate": 0.0,
                "distance": self.distance(serving_uav, ue),
                "path_loss_db": self.path_loss(serving_uav, ue),
            }

        sinr_values = []
        rate = 0.0
        for rb_idx in assigned_rbs:
            sinr = self.compute_rb_sinr(
                serving_uav=serving_uav,
                ue=ue,
                rb_idx=rb_idx,
                serving_assignment=serving_assignment,
                interfering_allocations=interfering_allocations,
            )
            sinr_values.append(sinr)
            rate += self.rb_bandwidth * math.log2(1.0 + sinr)

        mean_sinr = sum(sinr_values) / len(sinr_values)
        return {
            "ue": ue,
            "assigned_rbs": assigned_rbs,
            "sinr_per_rb": sinr_values,
            "mean_sinr": mean_sinr,
            "rate": rate,
            "distance": self.distance(serving_uav, ue),
            "path_loss_db": self.path_loss(serving_uav, ue),
        }

    def compute_assignment_metrics(
        self,
        serving_uav,
        ue_list: List,
        serving_assignment: Dict[int, List[int]],
        interfering_allocations: Optional[Iterable[Tuple[object, Dict[int, List[int]]]]] = None,
    ) -> List[dict]:
        """计算一个 UAV 的 OFDMA 分配结果。"""
        covered = self.find_covered_ues(serving_uav, ue_list)
        covered_map = {ue.uid: ue for ue in covered}
        results = []
        for ue_uid, rb_list in serving_assignment.items():
            ue = covered_map.get(ue_uid)
            if ue is None or not rb_list:
                continue
            metrics = self.compute_user_rate(
                serving_uav=serving_uav,
                ue=ue,
                serving_assignment=serving_assignment,
                interfering_allocations=interfering_allocations,
            )
            results.append(metrics)
        return results
