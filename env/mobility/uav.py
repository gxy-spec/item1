import numpy as np
from typing import Optional, Tuple


class UAV:
    """
    三维无人机（UAV）运动模型，支持约束随机运动。

    该类实现了UAV的三维位置更新，包括速度限制、高度约束和平面边界约束。
    运动采用随机扰动的方式，确保满足最大速度、高度范围和服务范围等约束。
    """

    def __init__(
        self,
        uid: int,
        position: np.ndarray,
        velocity: np.ndarray,
        vmax: float,
        hmin: float,
        hmax: float,
        service_radius: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        初始化UAV实例。

        Args:
            uid: UAV的唯一标识符。
            position: 初始三维位置 [x, y, h]。
            velocity: 初始三维速度 [vx, vy, vz]。
            vmax: 最大允许速度（标量）。
            hmin: 最小飞行高度。
            hmax: 最大飞行高度。
            service_radius: 地面服务覆盖半径。
            bounds: 可选的平面运动边界 (xmin, xmax, ymin, ymax)。
        """
        self.uid = uid
        self.position = np.asarray(position, dtype=float)  # 当前位置向量
        self.velocity = np.asarray(velocity, dtype=float)  # 当前速度向量
        self.vmax = float(vmax)  # 最大速度限制
        self.hmin = float(hmin)  # 最小高度限制
        self.hmax = float(hmax)  # 最大高度限制
        self.service_radius = float(service_radius)  # 服务半径
        self.bounds = bounds  # 平面边界约束

        # 初始化时强制执行约束
        self._enforce_velocity_limit()
        self._enforce_height_bounds()

    def _enforce_velocity_limit(self) -> None:
        """
        强制执行速度限制。

        如果当前速度超过 vmax，则将其缩放到 vmax。
        """
        speed = np.linalg.norm(self.velocity)  # 计算当前速度大小
        if speed > self.vmax and speed > 0:
            self.velocity = self.velocity * (self.vmax / speed)  # 归一化到 vmax

    def _enforce_height_bounds(self) -> None:
        """
        强制执行高度约束。

        将高度限制在 [hmin, hmax] 范围内，并调整垂直速度以避免越界。
        """
        self.position[2] = np.clip(self.position[2], self.hmin, self.hmax)  # 裁剪高度
        # 如果在最低高度且向下运动，反转垂直速度
        if self.position[2] <= self.hmin and self.velocity[2] < 0:
            self.velocity[2] = 0.0  # 停止向下运动
        # 如果在最高高度且向上运动，反转垂直速度
        if self.position[2] >= self.hmax and self.velocity[2] > 0:
            self.velocity[2] = 0.0  # 停止向上运动

    def randomize_velocity(self) -> None:
        """
        生成新的随机速度方向。

        在当前速度基础上添加高斯噪声，然后重新应用约束。
        """
        dv = np.random.normal(loc=0.0, scale=self.vmax * 0.3, size=3)  # 添加随机扰动
        self.velocity += dv
        self._enforce_velocity_limit()  # 重新应用速度限制

        # 根据高度约束调整垂直速度
        if self.position[2] <= self.hmin and self.velocity[2] < 0:
            self.velocity[2] = abs(self.velocity[2])  # 反转为向上
        if self.position[2] >= self.hmax and self.velocity[2] > 0:
            self.velocity[2] = -abs(self.velocity[2])  # 反转为向下

    def step(self, delta_t: float) -> None:
        """
        执行一步位置更新。

        使用当前速度更新位置，并应用所有约束。

        Args:
            delta_t: 时间步长。
        """
        self.randomize_velocity()  # 随机化速度
        self.position += self.velocity * float(delta_t)  # 更新位置

        self._enforce_height_bounds()  # 应用高度约束
        self._enforce_planar_bounds()  # 应用平面边界约束

    def _enforce_planar_bounds(self) -> None:
        """
        强制执行平面边界约束。

        如果定义了边界，将位置限制在范围内，并反转相应速度分量。
        """
        if self.bounds is None:
            return

        xmin, xmax, ymin, ymax = self.bounds
        x, y, h = self.position

        # 处理x方向边界
        if x < xmin:
            x = xmin
            self.velocity[0] = abs(self.velocity[0])  # 反转为向右
        elif x > xmax:
            x = xmax
            self.velocity[0] = -abs(self.velocity[0])  # 反转为向左

        # 处理y方向边界
        if y < ymin:
            y = ymin
            self.velocity[1] = abs(self.velocity[1])  # 反转为向上
        elif y > ymax:
            y = ymax
            self.velocity[1] = -abs(self.velocity[1])  # 反转为向下

        self.position[0] = x
        self.position[1] = y

    @property
    def xy(self) -> np.ndarray:
        """
        获取UAV的地面投影位置。

        Returns:
            二维数组 [x, y]。
        """
        return self.position[:2]

    @property
    def height(self) -> float:
        """
        获取UAV的高度。

        Returns:
            当前高度值。
        """
        return float(self.position[2])

    def __repr__(self) -> str:
        """
        返回UAV的字符串表示。

        Returns:
            描述UAV状态的字符串。
        """
        return (
            f"UAV(uid={self.uid}, position={self.position.tolist()}, "
            f"velocity={self.velocity.tolist()}, service_radius={self.service_radius})"
        )
