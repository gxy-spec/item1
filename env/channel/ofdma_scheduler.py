import math
from typing import Dict, List


class OFDMAScheduler:
    """最小版 OFDMA RB 调度器。

    当前先提供三种分配方式：
    - round_robin: 逐 RB 轮转分配
    - equal_share: 尽量平均切块分配
    - aoi_weighted: 按 AoI 权重比例分配
    """

    @staticmethod
    def _covered_ues(channel, uav, ue_list: List) -> List:
        return channel.find_covered_ues(uav, ue_list)

    @staticmethod
    def round_robin(channel, uav, ue_list: List) -> Dict[int, List[int]]:
        covered = OFDMAScheduler._covered_ues(channel, uav, ue_list)
        if not covered:
            return {}

        assignment = {ue.uid: [] for ue in covered}
        for rb_idx in range(channel.num_rbs):
            ue = covered[rb_idx % len(covered)]
            assignment[ue.uid].append(rb_idx)
        return assignment

    @staticmethod
    def equal_share(channel, uav, ue_list: List) -> Dict[int, List[int]]:
        covered = OFDMAScheduler._covered_ues(channel, uav, ue_list)
        if not covered:
            return {}

        assignment = {ue.uid: [] for ue in covered}
        n_users = len(covered)
        base = channel.num_rbs // n_users
        extra = channel.num_rbs % n_users

        rb_cursor = 0
        for idx, ue in enumerate(covered):
            count = base + (1 if idx < extra else 0)
            for _ in range(count):
                assignment[ue.uid].append(rb_cursor)
                rb_cursor += 1
        return assignment

    @staticmethod
    def aoi_weighted(channel, uav, ue_list: List, aoi_map: Dict[int, float]) -> Dict[int, List[int]]:
        covered = OFDMAScheduler._covered_ues(channel, uav, ue_list)
        if not covered:
            return {}

        assignment = {ue.uid: [] for ue in covered}
        weights = []
        for ue in covered:
            weights.append(max(float(aoi_map.get(ue.uid, 1.0)), 1e-6))

        weight_sum = sum(weights)
        raw_quota = [channel.num_rbs * w / weight_sum for w in weights]
        quota = [int(math.floor(item)) for item in raw_quota]
        allocated = sum(quota)

        remainders = sorted(
            [(raw_quota[idx] - quota[idx], idx) for idx in range(len(covered))],
            reverse=True,
        )
        for _, idx in remainders[: channel.num_rbs - allocated]:
            quota[idx] += 1

        rb_cursor = 0
        for idx, ue in enumerate(covered):
            for _ in range(quota[idx]):
                if rb_cursor >= channel.num_rbs:
                    break
                assignment[ue.uid].append(rb_cursor)
                rb_cursor += 1

        return assignment
