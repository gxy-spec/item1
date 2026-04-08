import numpy as np
from typing import Tuple


class UE:
    """
    二维地面用户设备（UE）随机游走模型。

    该类实现了UE在平面上的随机游走运动，位置被限制在指定的边界范围内。
    """

    def __init__(
        self,
        uid: int,
        position: np.ndarray,
        speed: float,
        bounds: Tuple[float, float, float, float],
    ):
        """
        初始化UE实例。

        Args:
            uid: UE的唯一标识符。
            position: 初始二维位置 [x, y]。
            speed: 随机游走的最大步长大小。
            bounds: 运动边界 (xmin, xmax, ymin, ymax)。
        """
        self.uid = uid
        self.position = np.asarray(position, dtype=float)  # 当前位置向量
        self.speed = float(speed)  # 最大步长
        self.bounds = bounds  # 边界约束

        self._enforce_bounds()  # 初始化时强制执行边界约束

    def _enforce_bounds(self) -> None:
        """
        强制执行边界约束。

        将位置限制在指定的矩形范围内。
        """
        xmin, xmax, ymin, ymax = self.bounds
        self.position[0] = np.clip(self.position[0], xmin, xmax)  # 裁剪x坐标
        self.position[1] = np.clip(self.position[1], ymin, ymax)  # 裁剪y坐标

    def step(self, delta_t: float) -> None:
        """
        执行一步随机游走。

        生成随机方向的位移，并更新位置，同时应用边界约束。

        Args:
            delta_t: 时间步长。
        """
        step_size = self.speed * float(delta_t)  # 计算步长
        direction = np.random.uniform(-1.0, 1.0, size=2)  # 随机方向向量
        if np.linalg.norm(direction) == 0:  # 避免零向量
            direction = np.array([1.0, 0.0])

        direction = direction / np.linalg.norm(direction)  # 归一化方向
        displacement = direction * step_size  # 计算位移
        self.position += displacement  # 更新位置
        self._enforce_bounds()  # 应用边界约束

    @property
    def xy(self) -> np.ndarray:
        """
        获取UE的位置。

        Returns:
            二维数组 [x, y]。
        """
        return self.position

    def __repr__(self) -> str:
        """
        返回UE的字符串表示。

        Returns:
            描述UE状态的字符串。
        """
        return f"UE(uid={self.uid}, position={self.position.tolist()})"
