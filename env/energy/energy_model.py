import math
import numpy as np
from typing import Dict, Tuple


class EnergyModel:
    """
    无人机能量模型（严格按照论文公式实现）。
    
    包含三部分能耗：
    1. 飞行能耗（P_fly）
    2. 通信能耗（E_tx）
    3. 无线充电（E_charge）
    
    电池动态方程：E(t+1) = E(t) - E_fly - E_tx + E_charge
    """

    def __init__(
        self,
        # 飞行能耗参数
        P0: float = 79.856,                    # 叶片剖面功率（W）
        Pi: float = 88.6,                      # 诱导功率（W）
        U_tip: float = 120.0,                  # 叶尖速度（m/s）
        v0: float = 9.2,                       # 悬停诱导速度（m/s）
        d0: float = 0.012,                     # 机身阻力比（无量纲）
        rho: float = 1.225,                    # 空气密度（kg/m^3）
        s: float = 0.05,                       # 转子实度（无量纲）
        A: float = 0.503,                      # 旋翼盘面积（m^2）
        # 通信能耗参数
        p_tx: float = 0.1,                     # 每个UE的发射功率（W）
        # 充电参数
        eta: float = 0.8,                      # 能量转换效率（无量纲）
        P_HAP: float = 300.0,                  # HAP发射功率（W）- 提高功率以增强充电效果
        # 电池参数
        E_max: float = 5000.0,                 # 最大电池容量（J）
        # 状态转换阈值
        return_threshold: float = 0.2,         # 返回充电的电量阈值（相对E_max）
        recovery_threshold: float = 0.8,       # 恢复正常运动的电量阈值（相对E_max）
        charging_distance: float = 30.0,       # 充电距离阈值（m）
    ):
        """
        初始化能量模型参数。
        
        Args:
            P0: 叶片剖面功率（W）
            Pi: 诱导功率（W）
            U_tip: 叶尖速度（m/s）
            v0: 悬停诱导速度（m/s）
            d0: 机身阻力比（无量纲，范围0.005-0.02）
            rho: 空气密度（kg/m^3，标准值1.225）
            s: 转子实度（无量纲，范围0.04-0.08）
            A: 旋翼盘面积（m^2）
            p_tx: 每个UE的发射功率（W）
            eta: 能量转换效率（无量纲，0.5-0.9）
            P_HAP: HAP发射功率（W）
            E_max: 最大电池容量（J）
            return_threshold: 返回充电的电量阈值（相对E_max）
            recovery_threshold: 恢复正常的电量阈值（相对E_max）
            charging_distance: 充电距离阈值（m）
        """
        # 飞行能耗参数
        self.P0 = float(P0)
        self.Pi = float(Pi)
        self.U_tip = float(U_tip)
        self.v0 = float(v0)
        self.d0 = float(d0)
        self.rho = float(rho)
        self.s = float(s)
        self.A = float(A)
        
        # 通信能耗参数
        self.p_tx = float(p_tx)
        
        # 充电参数
        self.eta = float(eta)
        self.P_HAP = float(P_HAP)
        
        # 电池参数
        self.E_max = float(E_max)
        self.return_threshold = float(return_threshold)
        self.recovery_threshold = float(recovery_threshold)
        self.charging_distance = float(charging_distance)

    # ========================================
    # 【第部分：飞行能耗模型】
    # ========================================
    # 公式：P_fly = P0 * (1 + 3*v^2 / U_tip^2)
    #              + Pi * sqrt(sqrt(1 + v^4 / (4*v0^4)) - v^2 / (2*v0^2))
    #              + 0.5 * d0 * rho * s * A * v^3
    # ========================================
    
    def flying_power(self, velocity: np.ndarray) -> float:
        """
        计算飞行功率（严格按照论文公式）。
        
        P_fly = P0 * (1 + 3*v^2 / U_tip^2)                              [叶片剖面功率]
              + Pi * sqrt(sqrt(1 + v^4 / (4*v0^4)) - v^2 / (2*v0^2))    [诱导功率]
              + 0.5 * d0 * rho * s * A * v^3                             [阻力功率]
        
        Args:
            velocity: 速度向量 [vx, vy, vz]（m/s）
        
        Returns:
            飞行功率 P_fly（W）
        """
        # 计算速度标量
        v = np.linalg.norm(velocity)
        
        # 避免数值问题
        if v < 1e-6:
            v = 1e-6
        
        # ============ 项1：叶片剖面功率 ============
        # P0_term = P0 * (1 + 3*v^2 / U_tip^2)
        v_squared = v ** 2
        U_tip_squared = self.U_tip ** 2
        P0_term = self.P0 * (1.0 + 3.0 * v_squared / U_tip_squared)
        
        # ============ 项2：诱导功率 ============
        # Pi_term = Pi * sqrt(sqrt(1 + v^4 / (4*v0^4)) - v^2 / (2*v0^2))
        v0_squared = self.v0 ** 2
        v0_fourth = v0_squared ** 2
        v_fourth = v_squared ** 2
        
        # 内部计算
        inner_fraction = v_fourth / (4.0 * v0_fourth)
        sqrt_term = math.sqrt(1.0 + inner_fraction)
        subtraction = sqrt_term - v_squared / (2.0 * v0_squared)
        
        # 确保不出现负数（数值稳定性）
        if subtraction < 0:
            subtraction = 1e-10
        
        Pi_term = self.Pi * math.sqrt(subtraction)
        
        # ============ 项3：阻力功率 ============
        # Drag_term = 0.5 * d0 * rho * s * A * v^3
        v_cubed = v ** 3
        drag_term = 0.5 * self.d0 * self.rho * self.s * self.A * v_cubed
        
        # ============ 总功率 ============
        P_fly = P0_term + Pi_term + drag_term
        
        return float(P_fly)

    def flying_energy(self, velocity: np.ndarray, delta_t: float) -> float:
        """
        计算飞行能耗。
        
        E_fly = P_fly * delta_t
        
        Args:
            velocity: 速度向量（m/s）
            delta_t: 时间步长（s）
        
        Returns:
            飞行能耗（J）
        """
        P_fly = self.flying_power(velocity)
        E_fly = P_fly * delta_t
        return float(E_fly)

    # ========================================
    # 【第二部分：通信能耗模型】
    # ========================================
    # 公式：E_tx = sum_{k ∈ K_n} p_k * delta_t
    # ========================================
    
    def tx_energy(self, num_covered_ue: int, delta_t: float) -> float:
        """
        计算通信能耗（发射功率 * 时间）。
        
        E_tx = sum_{k ∈ K_n} p_k * delta_t
        
        其中 K_n 为该UAV服务范围内的所有UE。
        
        Args:
            num_covered_ue: UAV服务范围内的UE数量
            delta_t: 时间步长（s）
        
        Returns:
            通信能耗（J）
        """
        # E_tx = p_tx * num_ue * delta_t
        E_tx = self.p_tx * num_covered_ue * delta_t
        return float(E_tx)

    # ========================================
    # 【第三部分：无线充电模型】
    # ========================================
    # 公式：E_charge = eta * P_HAP * g_nH * delta_t
    # ========================================
    
    def charging_energy(self, channel_gain_nH: float, delta_t: float) -> float:
        """
        计算充电能量（来自HAP的无线能量传输）。
        
        E_charge = eta * P_HAP * g_nH * delta_t
        
        其中 g_nH 是UAV到HAP的信道增益（必须从A2A信道模型动态获取）。
        
        Args:
            channel_gain_nH: UAV到HAP的信道增益（幅度，非dB）
            delta_t: 时间步长（s）
        
        Returns:
            充电能量（J）
        """
        # E_charge = eta * P_HAP * g_nH * delta_t
        E_charge = self.eta * self.P_HAP * channel_gain_nH * delta_t
        return float(E_charge)

    # ========================================
    # 【第四部分：电池动态方程】
    # ========================================
    # 公式：E_n(t+1) = E_n(t) - E_fly - E_tx + E_charge
    #      约束：0 ≤ E_n(t) ≤ E_max
    # ========================================
    
    def update_battery(
        self,
        current_energy: float,
        E_fly: float,
        E_tx: float,
        E_charge: float,
    ) -> float:
        """
        更新电池容量（电池动态方程）。
        
        E_n(t+1) = E_n(t) - E_fly - E_tx + E_charge
        受约束：0 ≤ E_n(t) ≤ E_max
        
        Args:
            current_energy: 当前电池容量（J）
            E_fly: 飞行能耗（J）
            E_tx: 通信能耗（J）
            E_charge: 充电能量（J）
        
        Returns:
            更新后的电池容量（J）
        """
        # 计算新能量
        new_energy = current_energy - E_fly - E_tx + E_charge
        
        # 强制执行约束
        new_energy = max(0.0, min(new_energy, self.E_max))
        
        return float(new_energy)

    # ========================================
    # 【附加方法：状态判断】
    # ========================================
    
    def get_energy_threshold(self) -> Tuple[float, float]:
        """
        获取电量阈值。
        
        Returns:
            (返回充电阈值, 恢复阈值) = (0.2*E_max, 0.9*E_max)
        """
        return (
            self.return_threshold * self.E_max,
            self.recovery_threshold * self.E_max,
        )

    def should_return_to_charging(self, current_energy: float) -> bool:
        """
        判断是否应该返回充电（电量过低）。
        
        Args:
            current_energy: 当前电量（J）
        
        Returns:
            True 如果 current_energy ≤ 0.2*E_max
        """
        return current_energy <= self.return_threshold * self.E_max

    def should_resume_normal(self, current_energy: float) -> bool:
        """
        判断是否可以恢复正常运动（电量充足）。
        
        Args:
            current_energy: 当前电量（J）
        
        Returns:
            True 如果 current_energy ≥ 0.9*E_max
        """
        return current_energy >= self.recovery_threshold * self.E_max

    # ========================================
    # 【调试方法：公式验证】
    # ========================================
    
    def log_power_breakdown(self, velocity: np.ndarray) -> Dict[str, float]:
        """
        计算并返回飞行功率的各部分分解（用于调试和可视化）。
        
        Returns:
            包含以下键的字典：
            - 'P0_term': 叶片剖面功率
            - 'Pi_term': 诱导功率
            - 'drag_term': 阻力功率
            - 'total': 总飞行功率
        """
        v = np.linalg.norm(velocity)
        
        if v < 1e-6:
            v = 1e-6
        
        # 项1
        P0_term = self.P0 * (1.0 + 3.0 * v ** 2 / self.U_tip ** 2)
        
        # 项2
        v0_sq = self.v0 ** 2
        v0_4 = v0_sq ** 2
        v_4 = v ** 4
        inner = math.sqrt(1.0 + v_4 / (4.0 * v0_4))
        subtraction = inner - v ** 2 / (2.0 * v0_sq)
        subtraction = max(subtraction, 1e-10)
        Pi_term = self.Pi * math.sqrt(subtraction)
        
        # 项3
        drag_term = 0.5 * self.d0 * self.rho * self.s * self.A * v ** 3
        
        return {
            'P0_term': float(P0_term),
            'Pi_term': float(Pi_term),
            'drag_term': float(drag_term),
            'total': float(P0_term + Pi_term + drag_term),
        }
