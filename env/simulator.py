import os
import sys
from typing import List, Optional, Tuple

# 确保从任意工作目录运行时都能导入本地 mobility 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from channel.a2g_channel import A2GChannel
from channel.a2a_channel import A2AChannel
from energy.energy_model import EnergyModel
from mobility.uav import UAV
from mobility.ue import UE


class HAP:
    """
    高空平台（HAP）类，固定在天空中的一点。

    HAP 不运动，仅作为参考点存在。
    """

    def __init__(self, position: Tuple[float, float, float]):
        """
        初始化HAP。

        Args:
            position: 三维位置 (x, y, h)。
        """
        self.position = np.asarray(position, dtype=float)  # 固定位置

    @property
    def xy(self) -> np.ndarray:
        """
        获取HAP的地面投影位置。

        Returns:
            二维数组 [x, y]。
        """
        return self.position[:2]

    @property
    def height(self) -> float:
        """
        获取HAP的高度。

        Returns:
            高度值。
        """
        return float(self.position[2])

    def __repr__(self) -> str:
        """
        返回HAP的字符串表示。

        Returns:
            描述HAP位置的字符串。
        """
        return f"HAP(position={self.position.tolist()})"


class Simulator:
    """
    仿真器类，负责协调UAV和UE的运动，并提供实时可视化。

    使用matplotlib动画展示节点运动过程。
    """

    def __init__(
        self,
        uav_list: List[UAV],
        ue_list: List[UE],
        hap: HAP,
        area_bounds: Tuple[float, float, float, float],
        a2g_channel: A2GChannel,
        a2a_channel: A2AChannel,
        delta_t: float = 0.5,
    ):
        """
        初始化仿真器。

        Args:
            uav_list: UAV对象列表。
            ue_list: UE对象列表。
            hap: HAP对象。
            area_bounds: 仿真区域边界 (xmin, xmax, ymin, ymax)。
            delta_t: 时间步长。
        """
        self.uav_list = uav_list  # UAV列表
        self.ue_list = ue_list    # UE列表
        self.hap = hap            # HAP对象
        self.area_bounds = area_bounds  # 区域边界
        self.delta_t = float(delta_t)   # 时间步长
        self.a2g_channel = a2g_channel
        self.a2a_channel = a2a_channel
        
        # ============ 能量模型初始化 ============
        self.energy_model = EnergyModel()
        
        # 为所有UAV设置初始电量和最大电量
        for uav in self.uav_list:
            uav.energy_max = self.energy_model.E_max
            uav.energy = self.energy_model.E_max  # 初始满电

        self.current_frame = 0
        self._snapshot_initial_state()

        # 初始化matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self._configure_axes()  # 配置坐标轴
        # 初始化艺术家对象（用于动画）
        self.uav_scatter = None
        self.ue_scatter = None
        self.hap_artist = None
        self.service_circles = []  # UAV服务范围圆圈
        self.altitude_text = None  # 显示高度信息的文本

        # 性能曲线历史数据
        self.time_history = []
        self.avg_a2g_sinr_history = []
        self.avg_a2a_snr_history = []
        self.avg_rate_history = []
        self.uav_sinr_history = {uav.uid: [] for uav in self.uav_list}  # 每个UAV的SINR历史
        self.uav_rate_history = {uav.uid: [] for uav in self.uav_list}  # 每个UAV的速率历史
        
        # ============ 能量历史数据 ============
        self.energy_history = {uav.uid: [] for uav in self.uav_list}  # 各UAV电量历史
        self.flying_energy_history = {uav.uid: [] for uav in self.uav_list}  # 飞行能耗
        self.tx_energy_history = {uav.uid: [] for uav in self.uav_list}  # 通信能耗
        self.charging_energy_history = {uav.uid: [] for uav in self.uav_list}  # 充电能量

        # ============ 位置历史数据（用于save模式） ============
        self.uav_position_history = []  # 每帧UAV位置列表
        self.ue_position_history = []   # 每帧UE位置列表
        self.energy_state_history = []  # 每帧能量状态
        self.metrics_history = []       # 每帧性能指标

        # 初始化热力图窗口
        self._init_heatmap_figure()
        
        # 初始化能量窗口
        self._init_energy_figure()

    def _snapshot_initial_state(self) -> None:
        """保存初始场景，用于reset。"""
        self.initial_uav_states = [
            {
                "position": uav.position.copy(),
                "velocity": uav.velocity.copy(),
                "energy": uav.energy,
                "energy_state": uav.energy_state,
            }
            for uav in self.uav_list
        ]
        self.initial_ue_positions = [ue.position.copy() for ue in self.ue_list]

    def _reset_histories(self) -> None:
        """清空历史缓存。"""
        self.time_history = []
        self.avg_a2g_sinr_history = []
        self.avg_a2a_snr_history = []
        self.avg_rate_history = []
        self.uav_sinr_history = {uav.uid: [] for uav in self.uav_list}
        self.uav_rate_history = {uav.uid: [] for uav in self.uav_list}
        self.energy_history = {uav.uid: [] for uav in self.uav_list}
        self.flying_energy_history = {uav.uid: [] for uav in self.uav_list}
        self.tx_energy_history = {uav.uid: [] for uav in self.uav_list}
        self.charging_energy_history = {uav.uid: [] for uav in self.uav_list}
        self.uav_position_history = []
        self.ue_position_history = []
        self.energy_state_history = []
        self.metrics_history = []

    def reset(self) -> dict:
        """恢复到初始状态，并返回当前观测。"""
        for uav, state in zip(self.uav_list, self.initial_uav_states):
            uav.position = state["position"].copy()
            uav.velocity = state["velocity"].copy()
            uav.energy = state["energy"]
            uav.energy_state = state["energy_state"]

        for ue, position in zip(self.ue_list, self.initial_ue_positions):
            ue.position = position.copy()

        self.current_frame = 0
        self._reset_histories()
        return self.get_observation()

    def get_observation(self) -> dict:
        """返回当前场景观测。"""
        return {
            "time": self.current_frame * self.delta_t,
            "uavs": [
                {
                    "uid": uav.uid,
                    "position": uav.position.copy(),
                    "velocity": uav.velocity.copy(),
                    "energy": float(uav.energy),
                    "energy_state": uav.energy_state,
                }
                for uav in self.uav_list
            ],
            "ues": [
                {
                    "uid": ue.uid,
                    "position": ue.position.copy(),
                }
                for ue in self.ue_list
            ],
            "hap": self.hap.position.copy(),
        }

    def _prepare_uav_motion(self) -> None:
        """依据当前状态为UAV准备本时隙动作。"""
        for uav in self.uav_list:
            if uav.energy_state == "depleted":
                uav.velocity[:] = 0.0
                continue

            if uav.energy_state not in {"return", "resume"}:
                continue

            if uav.energy_state == "return":
                target = self._get_charging_waypoint(uav)
                arrival_state = "charging"
            else:
                target = uav.home_position
                arrival_state = "normal"

            diff = target - uav.position
            distance = np.linalg.norm(diff)
            if distance < self.energy_model.charging_distance:
                uav.energy_state = arrival_state
                uav.velocity[:] = 0.0
                continue

            direction = diff / (distance + 1e-6)
            uav.velocity = direction * uav.vmax

    def _get_charging_waypoint(self, uav: UAV) -> np.ndarray:
        """返回UAV可实际到达的充电等待点。"""
        charging_height = min(uav.hmax, max(uav.hmin, self.hap.height - 300.0))
        return np.array([self.hap.xy[0], self.hap.xy[1], charging_height], dtype=float)

    def _estimate_return_energy_budget(self, uav: UAV) -> float:
        """估计UAV安全返航到充电点所需的最低能量预算。"""
        target = self._get_charging_waypoint(uav)
        distance = np.linalg.norm(target - uav.position)
        travel_time = distance / max(uav.vmax, 1e-6)
        cruise_velocity = target - uav.position
        norm = np.linalg.norm(cruise_velocity)
        if norm > 1e-6:
            cruise_velocity = cruise_velocity / norm * uav.vmax
        else:
            cruise_velocity = np.zeros(3, dtype=float)

        travel_energy = self.energy_model.flying_energy(cruise_velocity, travel_time)
        reserve_energy = 0.05 * self.energy_model.E_max
        return min(self.energy_model.E_max, travel_energy + reserve_energy)

    def _advance_entities(self) -> None:
        """推进一个时隙内的节点运动。"""
        self._prepare_uav_motion()
        for uav in self.uav_list:
            uav.step(self.delta_t)
        for ue in self.ue_list:
            ue.step(self.delta_t)

    def step(self) -> Tuple[dict, dict]:
        """推进一个时隙，并返回观测与性能指标。"""
        frame = self.current_frame
        self._advance_entities()
        metrics = self._compute_channel_metrics(frame)
        self._update_energy(metrics)
        self.current_frame += 1
        return self.get_observation(), metrics

    def _configure_axes(self) -> None:
        """
        配置matplotlib坐标轴。

        设置轴范围、标题、标签和网格。
        """
        xmin, xmax, ymin, ymax = self.area_bounds
        self.ax.set_xlim(xmin, xmax)  # 设置x轴范围
        self.ax.set_ylim(ymin, ymax)  # 设置y轴范围
        self.ax.set_title("UAV / UE Mobility Simulation")  # 标题
        self.ax.set_xlabel("X position")  # x轴标签
        self.ax.set_ylabel("Y position")  # y轴标签
        self.ax.grid(True, linestyle="--", alpha=0.5)  # 网格

    def _init_performance_figure(self) -> None:
        """初始化性能窗口，展示平均A2G SINR、A2A SNR和速率随时间的变化。"""
        self.fig2, (self.ax_sinr, self.ax_snr, self.ax_rate) = plt.subplots(
            3,
            1,
            figsize=(9, 11),
            sharex=True,
        )
        self.fig2.suptitle("Channel Performance Metrics", fontsize=14)

        # 初始化平均值曲线
        self.ax_sinr.set_ylabel("A2G SINR (dB)")
        self.ax_sinr.grid(True, linestyle="--", alpha=0.4)
        self.avg_a2g_line, = self.ax_sinr.plot([], [], c="tab:blue", label="Avg A2G SINR", linewidth=2, linestyle='--')
        
        # 初始化每个UAV的SINR曲线
        self.uav_sinr_lines = {}
        colors = ['tab:red', 'tab:purple', 'tab:brown']
        for i, uav in enumerate(self.uav_list):
            line, = self.ax_sinr.plot([], [], c=colors[i % len(colors)], 
                                    label=f"UAV{uav.uid} SINR", linewidth=1.5, alpha=0.8)
            self.uav_sinr_lines[uav.uid] = line
        
        self.ax_sinr.legend(loc="upper right")

        self.ax_snr.set_ylabel("A2A SNR (dB)")
        self.ax_snr.grid(True, linestyle="--", alpha=0.4)
        self.avg_a2a_line, = self.ax_snr.plot([], [], c="tab:orange", label="Avg A2A SNR", linewidth=2)
        self.ax_snr.legend(loc="upper right")

        self.ax_rate.set_ylabel("Rate (bps)")
        self.ax_rate.set_xlabel("Time (s)")
        self.ax_rate.grid(True, linestyle="--", alpha=0.4)
        self.avg_rate_line, = self.ax_rate.plot([], [], c="tab:green", label="Avg Rate", linewidth=2, linestyle='--')
        
        # 初始化每个UAV的速率曲线
        self.uav_rate_lines = {}
        for i, uav in enumerate(self.uav_list):
            line, = self.ax_rate.plot([], [], c=colors[i % len(colors)], 
                                    label=f"UAV{uav.uid} Rate", linewidth=1.5, alpha=0.8)
            self.uav_rate_lines[uav.uid] = line
        
        self.ax_rate.legend(loc="upper right")
        self.fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

    def _init_heatmap_figure(self) -> None:
        """初始化热力图窗口，显示多UAV的SINR热力图。"""
        self.fig3, self.ax_heatmap = plt.subplots(figsize=(10, 8))
        self.fig3.suptitle("Multi-UAV SINR Heatmap", fontsize=14)

        # 初始化热力图
        xx, yy, sinr_map = self.compute_uav_heatmap()
        
        # 对SINR值的转换：使用相对分贝值，突出显示差异
        valid_sinr = sinr_map[~np.isnan(sinr_map)]
        if len(valid_sinr) > 0:
            sinr_min = np.percentile(valid_sinr, 5)   # 5分位数
            sinr_max = np.percentile(valid_sinr, 95)  # 95分位数
            sinr_range = sinr_max - sinr_min
            sinr_min = max(sinr_min - 0.5 * sinr_range, 1e-10)
            sinr_max = sinr_max + 0.5 * sinr_range
        else:
            sinr_min, sinr_max = 1e-10, 0.2

        sinr_db = 10 * np.log10(np.maximum(sinr_map, 1e-10))
        
        self.heatmap_im = self.ax_heatmap.imshow(
            sinr_db.T,
            extent=[yy.min(), yy.max(), xx.min(), xx.max()],  # 修正：转置后x,y轴交换
            origin='lower',
            cmap='turbo',  # 更能显示细微差别
            aspect='auto',
            vmin=10 * np.log10(max(sinr_min, 1e-10)),
            vmax=10 * np.log10(max(sinr_max, 1e-10)),
        )
        self.ax_heatmap.set_xlabel("X position (m)")
        self.ax_heatmap.set_ylabel("Y position (m)")
        self.ax_heatmap.set_title("Multi-UAV SINR Distribution (dB)")

        # 添加颜色条
        self.cbar = self.fig3.colorbar(self.heatmap_im, ax=self.ax_heatmap, label="SINR (dB)")

        # 绘制UAV和UE位置
        uav_positions = np.array([uav.xy for uav in self.uav_list])
        ue_positions = np.array([ue.xy for ue in self.ue_list])

        self.heatmap_uav_scatter = self.ax_heatmap.scatter(
            uav_positions[:, 0], uav_positions[:, 1],
            c="red", marker="^", s=200, label="UAV", zorder=5, edgecolors='darkred', linewidths=2
        )
        self.heatmap_ue_scatter = self.ax_heatmap.scatter(
            ue_positions[:, 0], ue_positions[:, 1],
            c="lime", marker="o", s=60, label="UE", zorder=4, edgecolors='darkgreen', linewidths=1
        )
        
        # 绘制UAV服务半径
        for uav in self.uav_list:
            circle = plt.Circle(
                tuple(uav.xy), uav.service_radius,
                edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2
            )
            self.ax_heatmap.add_patch(circle)
        
        self.ax_heatmap.legend(loc="upper right", fontsize=10)
        self.fig3.tight_layout()

    def _init_energy_figure(self) -> None:
        """
        初始化能量窗口，展示各UAV的能量状态。
        
        包含两个子图：
        1. 电量曲线（各UAV的E(t)）
        2. 能量分解图（能耗和充电分量）
        """
        self.fig4, (self.ax_energy, self.ax_energy_breakdown) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        self.fig4.suptitle("UAV Energy Management", fontsize=14)
        
        # 子图1：电量曲线
        self.ax_energy.set_ylabel("Energy (J)")
        self.ax_energy.set_title("UAV Battery Capacity Over Time")
        self.ax_energy.grid(True, linestyle="--", alpha=0.4)
        self.ax_energy.set_ylim(0, self.energy_model.E_max * 1.1)
        
        self.energy_lines = {}
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        for i, uav in enumerate(self.uav_list):
            color = colors[i % len(colors)]
            line, = self.ax_energy.plot(
                [], [], c=color, linewidth=2.5,
                label=f"UAV{uav.uid} Energy", marker='o', markersize=2, alpha=0.8
            )
            self.energy_lines[uav.uid] = line
        
        self.ax_energy.legend(loc="upper right", fontsize=10)
        
        # 子图2：能量分解
        self.ax_energy_breakdown.set_ylabel("Energy (J)")
        self.ax_energy_breakdown.set_xlabel("Time (s)")
        self.ax_energy_breakdown.set_title("Energy Components (Flying + TX - Charging)")
        self.ax_energy_breakdown.grid(True, linestyle="--", alpha=0.4)
        
        self.flying_energy_line, = self.ax_energy_breakdown.plot([], [], c='orange', label='Flying Energy', linewidth=2.5, alpha=0.8)
        self.tx_energy_line, = self.ax_energy_breakdown.plot([], [], c='red', label='TX Energy (x10)', linewidth=2.5, alpha=0.8)
        self.charging_energy_line, = self.ax_energy_breakdown.plot([], [], c='green', label='Charging Energy', linewidth=2.5, alpha=0.8)
        self.ax_energy_breakdown.legend(loc="upper right", fontsize=10)
        
        self.fig4.tight_layout()


    def _init_artists(self) -> List:
        """
        初始化动画艺术家对象。

        创建散点图、服务圆圈和文本，用于动画显示。

        Returns:
            艺术家对象列表，用于FuncAnimation。
        """
        # 获取初始位置
        uav_positions = np.array([uav.xy for uav in self.uav_list])
        ue_positions = np.array([ue.xy for ue in self.ue_list])

        # 创建UAV散点图（红色三角形）
        self.uav_scatter = self.ax.scatter(
            uav_positions[:, 0],
            uav_positions[:, 1],
            c="red",
            marker="^",
            s=120,
            label="UAV",
            zorder=3,
        )

        # 创建UE散点图（蓝色圆点）
        self.ue_scatter = self.ax.scatter(
            ue_positions[:, 0],
            ue_positions[:, 1],
            c="blue",
            marker="o",
            s=40,
            label="UE",
            zorder=2,
        )

        # 创建HAP散点图（绿色星形）
        self.hap_artist = self.ax.scatter(
            [self.hap.xy[0]],
            [self.hap.xy[1]],
            c="green",
            marker="*",
            s=220,
            label="HAP",
            zorder=4,
        )

        # 为每个UAV创建服务范围圆圈（虚线）
        for uav in self.uav_list:
            circle = plt.Circle(
                tuple(uav.xy),  # 圆心位置
                uav.service_radius,  # 半径
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                zorder=1,
            )
            self.ax.add_patch(circle)  # 添加到轴上
            self.service_circles.append(circle)

        # 添加图例
        self.ax.legend(loc="upper right")

        # 创建高度显示文本
        self.altitude_text = self.ax.text(
            0.02,  # x位置（相对坐标）
            0.95,  # y位置（相对坐标）
            "",    # 初始文本为空
            transform=self.ax.transAxes,  # 使用轴坐标系
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),  # 背景框
        )

        # 返回所有艺术家对象
        return [self.uav_scatter, self.ue_scatter, self.hap_artist, *self.service_circles, self.altitude_text]

    def _update_artists(self, frame: int) -> List:
        """
        更新动画艺术家对象（每帧调用）。

        更新所有节点位置，刷新显示。

        Args:
            frame: 当前帧号。

        Returns:
            需要更新的艺术家对象列表。
        """
        _, metrics = self.step()

        # 获取新位置
        uav_positions = np.array([uav.xy for uav in self.uav_list])
        ue_positions = np.array([ue.xy for ue in self.ue_list])

        # 更新散点图位置
        self.uav_scatter.set_offsets(uav_positions)
        self.ue_scatter.set_offsets(ue_positions)

        # 更新服务圆圈位置
        for circle, uav in zip(self.service_circles, self.uav_list):
            circle.center = tuple(uav.xy)

        # 更新高度显示文本与当前性能指标
        # 显示格式更清晰
        time_str = f"Time: {self.time_history[-1]:.1f}s"
        altitudes = " | ".join([f"UAV{uav.uid}: {uav.height:.0f}m" for uav in self.uav_list])
        uav_energy_text = " | ".join([f"E{uav.uid}: {uav.energy:.0f}J ({uav.energy_state})" for uav in self.uav_list])
        uav_sinr_text = " | ".join([f"SINR{uav.uid}: {metrics['uav_sinrs'][uav.uid]:.2f}" for uav in self.uav_list])
        uav_rate_text = " | ".join([f"R{uav.uid}: {metrics['uav_rates'][uav.uid] / 1e6:.1f}M" for uav in self.uav_list])
        self.altitude_text.set_text(
            f"{time_str}  |  HAP: {self.hap.height:.0f}m\nAlt: {altitudes}\n{uav_energy_text}\n{uav_sinr_text}\n{uav_rate_text} Mbps"
        )

        # 更新能量窗口
        self._update_energy_plot()
        self._update_performance_plot()

        # 返回需要更新的艺术家对象（只返回主窗口的艺术家）
        return [self.uav_scatter, self.ue_scatter, *self.service_circles, self.altitude_text]

    def _compute_channel_metrics(self, frame: int) -> dict:
        """计算当前帧的A2G和A2A性能指标。"""
        all_sinrs = []
        all_rates = []
        a2a_snrs = []
        uav_sinrs = {}
        uav_rates = {}

        for uav in self.uav_list:
            a2g_results = self.a2g_channel.compute_sinr_and_rate(uav, self.ue_list, self.uav_list)
            uav_sinrs[uav.uid] = float(np.mean([item["sinr"] for item in a2g_results])) if a2g_results else float('nan')
            uav_rates[uav.uid] = float(np.mean([item["rate"] for item in a2g_results])) if a2g_results else float('nan')
            all_sinrs.extend([item["sinr"] for item in a2g_results])
            all_rates.extend([item["rate"] for item in a2g_results])
            a2a_metrics = self.a2a_channel.compute_link_metrics(uav, self.hap)
            a2a_snrs.append(a2a_metrics["snr"])

        avg_a2g_sinr = float(np.mean(all_sinrs)) if len(all_sinrs) else float('nan')
        avg_rate = float(np.mean(all_rates)) if len(all_rates) else float('nan')
        avg_a2a_snr = float(np.mean(a2a_snrs)) if len(a2a_snrs) else float('nan')

        self.time_history.append(frame * self.delta_t)
        self.avg_a2g_sinr_history.append(avg_a2g_sinr)
        self.avg_a2a_snr_history.append(avg_a2a_snr)
        self.avg_rate_history.append(avg_rate)
        
        # 保存每个UAV的历史数据
        for uav in self.uav_list:
            self.uav_sinr_history[uav.uid].append(uav_sinrs[uav.uid])
            self.uav_rate_history[uav.uid].append(uav_rates[uav.uid])

        return {
            "avg_a2g_sinr": avg_a2g_sinr,
            "avg_a2a_snr": avg_a2a_snr,
            "avg_rate": avg_rate,
            "uav_sinrs": uav_sinrs,
            "uav_rates": uav_rates,
        }
    
    # ========================================
    # 【能量模型集成方法】
    # ========================================
    
    def _update_energy(self, metrics: dict) -> None:
        """
        计算和更新所有UAV的能量。
        
        对每个UAV执行以下步骤：
        1. 计算飞行能耗 E_fly = P_fly * delta_t
        2. 计算通信能耗 E_tx = p_tx * num_ue * delta_t
        3. 从A2A信道获取 g_nH
        4. 计算充电能量 E_charge = eta * P_HAP * g_nH * delta_t
        5. 更新电池容量 E(t+1) = E(t) - E_fly - E_tx + E_charge
        6. 实现状态机逻辑
        
        Args:
            metrics: 包含SINR、速率等指标的字典
        """
        for uav in self.uav_list:
            a2a_metrics = self.a2a_channel.compute_link_metrics(uav, self.hap)
            channel_gain_nH = a2a_metrics["gain"]
            return_energy_budget = self._estimate_return_energy_budget(uav)

            # ========== 步骤1/2：根据状态计算飞行和通信能耗 ==========
            if uav.energy_state == "charging":
                # 充电状态时停靠，不再消耗飞行和通信功率
                E_fly = 0.0
                E_tx = 0.0
                E_charge = self.energy_model.charging_energy(channel_gain_nH, self.delta_t)

                # 停靠充电时提供一个更高的保底功率，避免长时间停滞在充电点。
                min_charging_power = 200.0
                min_E_charge = min_charging_power * self.delta_t
                E_charge = max(E_charge, min_E_charge)
            elif uav.energy_state == "depleted":
                E_fly = 0.0
                E_tx = 0.0
                E_charge = 0.0
            else:
                E_fly = self.energy_model.flying_energy(uav.velocity, self.delta_t)
                covered_ues = self.a2g_channel.find_covered_ues(uav, self.ue_list)
                num_covered = len(covered_ues)
                E_tx = self.energy_model.tx_energy(num_covered, self.delta_t)
                E_charge = self.energy_model.charging_energy(channel_gain_nH, self.delta_t)
            
            # ========== 步骤5：更新电池容量 ==========
            uav.energy = self.energy_model.update_battery(
                uav.energy,
                E_fly,
                E_tx,
                E_charge,
            )

            charging_waypoint = self._get_charging_waypoint(uav)
            distance_to_charge = np.linalg.norm(charging_waypoint - uav.position)

            if uav.energy <= 0.0 and uav.energy_state not in {"charging", "resume"}:
                if distance_to_charge < self.energy_model.charging_distance:
                    uav.energy_state = "charging"
                else:
                    uav.energy_state = "depleted"
                    uav.velocity[:] = 0.0
                    self.energy_history[uav.uid].append(uav.energy)
                    self.flying_energy_history[uav.uid].append(E_fly)
                    self.tx_energy_history[uav.uid].append(E_tx)
                    self.charging_energy_history[uav.uid].append(E_charge)
                    continue
             
            # ========== 步骤6a：状态机逻辑 - 判断是否需要返回充电 ==========
            dynamic_return_threshold = max(
                self.energy_model.return_threshold * self.energy_model.E_max,
                return_energy_budget,
            )
            if uav.energy <= dynamic_return_threshold:
                if uav.energy_state not in {"charging", "depleted", "return"}:
                    uav.energy_state = "return"
             
            # ========== 步骤6b：处理"return"状态 - 飞向HAP ==========
            if uav.energy_state == "return":
                if distance_to_charge < self.energy_model.charging_distance:
                    uav.energy_state = "charging"
             
            # ========== 步骤6c：处理"charging"状态 - 停止运动 ==========
            elif uav.energy_state == "charging":
                # 停止水平运动，垂直速度为0
                uav.velocity[:] = 0.0
                if self.energy_model.should_resume_normal(uav.energy):
                    uav.energy_state = "resume"
            
            # ========== 步骤6d：处理"resume"状态 - 离开充电点恢复任务 ==========
            elif uav.energy_state == "resume":
                if np.linalg.norm(uav.home_position - uav.position) < self.energy_model.charging_distance:
                    uav.energy_state = "normal"
             
            # ========== 步骤6e：处理"normal"状态 - 正常随机运动 ==========
            elif uav.energy_state == "normal":
                # 正常运动由uav.step()处理
                pass
            
            # ========== 记录能量历史 ==========
            self.energy_history[uav.uid].append(uav.energy)
            self.flying_energy_history[uav.uid].append(E_fly)
            self.tx_energy_history[uav.uid].append(E_tx)
            self.charging_energy_history[uav.uid].append(E_charge)

    def _update_performance_plot(self) -> None:
        """刷新性能曲线图数据。"""
        x = np.array(self.time_history)
        self.avg_a2g_line.set_data(x, np.array(self.avg_a2g_sinr_history))
        self.avg_a2a_line.set_data(x, np.array(self.avg_a2a_snr_history))
        self.avg_rate_line.set_data(x, np.array(self.avg_rate_history))
        
        # 更新每个UAV的曲线
        for uav in self.uav_list:
            if uav.uid in self.uav_sinr_history and uav.uid in self.uav_sinr_lines:
                self.uav_sinr_lines[uav.uid].set_data(x, np.array(self.uav_sinr_history[uav.uid]))
            if uav.uid in self.uav_rate_history and uav.uid in self.uav_rate_lines:
                self.uav_rate_lines[uav.uid].set_data(x, np.array(self.uav_rate_history[uav.uid]))

        for ax in [self.ax_sinr, self.ax_snr, self.ax_rate]:
            ax.relim()
            ax.autoscale_view()

        if self.fig2 is not None:
            self.fig2.canvas.draw_idle()

    def _update_energy_plot(self) -> None:
        """刷新能量窗口数据。"""
        # 检查是否有数据
        if len(self.time_history) == 0:
            return
        
        x = np.array(self.time_history)
        
        # ========== 更新电量曲线 ==========
        for uav in self.uav_list:
            if uav.uid in self.energy_history and len(self.energy_history[uav.uid]) > 0:
                # 确保数据长度匹配
                energy_data = np.array(self.energy_history[uav.uid])
                if len(energy_data) > len(x):
                    energy_data = energy_data[:len(x)]
                elif len(energy_data) < len(x):
                    # 如果能量数据较短，用最后一个值填充
                    last_value = energy_data[-1] if len(energy_data) > 0 else 1000.0
                    padding = np.full(len(x) - len(energy_data), last_value)
                    energy_data = np.concatenate([energy_data, padding])
                
                self.energy_lines[uav.uid].set_data(x, energy_data)
        
        # 重新计算自动缩放
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
        # ========== 更新能量分解（显示所有UAV的平均值）==========
        if len(self.uav_list) > 0 and len(x) > 0:
            # 计算所有UAV的平均能量值
            all_flying = []
            all_tx = []
            all_charging = []
            
            for uav in self.uav_list:
                flying_hist = self.flying_energy_history.get(uav.uid, [])
                tx_hist = self.tx_energy_history.get(uav.uid, [])
                charging_hist = self.charging_energy_history.get(uav.uid, [])
                
                # 确保数据长度一致
                min_len = min(len(flying_hist), len(tx_hist), len(charging_hist), len(x))
                if min_len > 0:
                    all_flying.append(np.array(flying_hist[:min_len]))
                    all_tx.append(np.array(tx_hist[:min_len]))
                    all_charging.append(np.array(charging_hist[:min_len]))
            
            # 计算平均值
            if all_flying:
                avg_flying = np.mean(np.array(all_flying), axis=0)
                self.flying_energy_line.set_data(x[:len(avg_flying)], avg_flying)
            
            if all_tx:
                avg_tx = np.mean(np.array(all_tx), axis=0)
                self.tx_energy_line.set_data(x[:len(avg_tx)], avg_tx * 10)  # 放大10倍显示
            
            if all_charging:
                avg_charging = np.mean(np.array(all_charging), axis=0)
                self.charging_energy_line.set_data(x[:len(avg_charging)], avg_charging)
            
            self.ax_energy_breakdown.relim()
            self.ax_energy_breakdown.autoscale_view()
        
        if self.fig4 is not None:
            self.fig4.canvas.draw_idle()

    def _update_heatmap(self) -> None:
        """更新热力图显示。"""
        # 更新多UAV热力图
        xx, yy, sinr_map = self.compute_uav_heatmap()
        
        # 动态调整颜色范围以显示当前帧的差异
        valid_sinr = sinr_map[~np.isnan(sinr_map)]
        if len(valid_sinr) > 0:
            sinr_min = np.percentile(valid_sinr, 5)
            sinr_max = np.percentile(valid_sinr, 95)
            sinr_range = sinr_max - sinr_min
            sinr_min = max(sinr_min - 0.5 * sinr_range, 1e-10)
            sinr_max = sinr_max + 0.5 * sinr_range
        else:
            sinr_min, sinr_max = 1e-10, 0.2
        
        sinr_db = 10 * np.log10(np.maximum(sinr_map, 1e-10))
        self.heatmap_im.set_data(sinr_db.T)
        self.heatmap_im.set_extent([yy.min(), yy.max(), xx.min(), xx.max()])  # 修正：转置后x,y轴交换
        self.heatmap_im.set_clim(vmin=10 * np.log10(max(sinr_min, 1e-10)), vmax=10 * np.log10(max(sinr_max, 1e-10)))

        # 更新UAV和UE位置
        uav_positions = np.array([uav.xy for uav in self.uav_list])
        ue_positions = np.array([ue.xy for ue in self.ue_list])

        self.heatmap_uav_scatter.set_offsets(uav_positions)
        self.heatmap_ue_scatter.set_offsets(ue_positions)

        if self.fig3 is not None:
            self.fig3.canvas.draw_idle()

    def compute_uav_heatmap(self, grid_size: int = 80):
        """生成包含所有UAV覆盖区和多UAV干扰的SINR热力图。"""
        xx, yy, sinr_map = self.a2g_channel.compute_heatmap(
            self.uav_list, self.ue_list, self.area_bounds, grid_size=grid_size
        )
        return xx, yy, sinr_map

    def run(self, frames: int = 400, interval: int = 100, save_path: Optional[str] = None) -> None:
        """
        启动动画循环。

        Args:
            frames: 总帧数。
            interval: 帧间隔（毫秒）。
            save_path: 如果提供，将动画保存为文件而不是实时显示。
        """
        self.reset()

        if save_path is not None:
            # 如果要保存为文件，使用非GUI后端
            plt.switch_backend('Agg')

            # 运行仿真以收集数据
            print(f"正在运行仿真 ({frames}帧)...")
            
            for _ in range(frames):
                _, metrics = self.step()
                uav_positions = [uav.position.copy() for uav in self.uav_list]
                ue_positions = [ue.position.copy() for ue in self.ue_list]
                energy_states = [(uav.uid, uav.energy_state, uav.energy) for uav in self.uav_list]

                self.uav_position_history.append(uav_positions)
                self.ue_position_history.append(ue_positions)
                self.energy_state_history.append(energy_states)
                self.metrics_history.append(metrics)

            actual_frames = min(
                frames,
                len(self.uav_position_history),
                len(self.ue_position_history),
                len(self.energy_state_history),
                len(self.metrics_history),
            )
            print(f"仿真完成 ({actual_frames}帧)")

            # 初始化窗口
            self._init_artists()
            self._init_energy_figure()
            self._init_performance_figure()

            # 创建动画，使用保存的历史数据
            def update_for_save(frame):
                # 从历史数据恢复状态
                uav_positions = self.uav_position_history[frame]
                ue_positions = self.ue_position_history[frame]
                energy_states = self.energy_state_history[frame]
                metrics = self.metrics_history[frame]
                
                # 恢复位置
                for uav, pos in zip(self.uav_list, uav_positions):
                    uav.position = pos
                for ue, pos in zip(self.ue_list, ue_positions):
                    ue.position = pos
                
                # 恢复能量状态（用于显示）
                for uid, state, energy in energy_states:
                    for uav in self.uav_list:
                        if uav.uid == uid:
                            uav.energy_state = state
                            uav.energy = energy
                            break

                # 更新散点图位置
                self.uav_scatter.set_offsets(np.array([uav.xy for uav in self.uav_list]))
                self.ue_scatter.set_offsets(np.array([ue.xy for ue in self.ue_list]))

                # 更新服务圆圈位置
                for circle, uav in zip(self.service_circles, self.uav_list):
                    circle.center = tuple(uav.xy)

                # 更新高度显示文本
                time_str = f"Time: {frame * self.delta_t:.1f}s"
                altitudes = " | ".join([f"UAV{uav.uid}: {uav.height:.0f}m" for uav in self.uav_list])
                uav_energy_text = " | ".join([f"E{uav.uid}: {uav.energy:.0f}J ({uav.energy_state})" for uav in self.uav_list])
                uav_sinr_text = " | ".join([f"SINR{uav.uid}: {metrics['uav_sinrs'][uav.uid]:.2f}" for uav in self.uav_list])
                uav_rate_text = " | ".join([f"R{uav.uid}: {metrics['uav_rates'][uav.uid] / 1e6:.1f}M" for uav in self.uav_list])
                self.altitude_text.set_text(
                    f"{time_str}  |  HAP: {self.hap.height:.0f}m\nAlt: {altitudes}\n{uav_energy_text}\n{uav_sinr_text}\n{uav_rate_text} Mbps"
                )

                # 更新能量图
                self._update_energy_plot()
                self._update_performance_plot()
                
                return [self.uav_scatter, self.ue_scatter, *self.service_circles, self.altitude_text]

            anim = animation.FuncAnimation(
                self.fig,
                update_for_save,
                frames=actual_frames,
                interval=interval,
                blit=False,
                repeat=False,
            )

            # 保存动画
            try:
                writer = animation.PillowWriter(fps=1000/interval)
                anim.save(save_path, writer=writer)
                print(f"运动仿真动画已保存到: {save_path}")
                
                # 保存能量管理图
                energy_path = save_path.replace('.gif', '_energy.png')
                self.fig4.savefig(energy_path, dpi=150, bbox_inches='tight')
                print(f"能量管理图已保存到: {energy_path}")

                # 保存信道性能图和SINR热力图
                self._update_performance_plot()
                perf_path = save_path.replace('.gif', '_performance.png')
                self.fig2.savefig(perf_path, dpi=150, bbox_inches='tight')
                print(f"信道性能图已保存到: {perf_path}")

                self._init_heatmap_figure()
                heatmap_path = save_path.replace('.gif', '_heatmap.png')
                self.fig3.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                print(f"SINR热力图已保存到: {heatmap_path}")

            except Exception as e:
                print(f"保存动画失败: {e}")
                print("请确保安装了必要的依赖: pip install pillow")

        else:
            # 实时显示模式
            try:
                # 尝试使用交互式后端
                plt.switch_backend('TkAgg')
                # 初始化所有窗口：运动仿真 + 能量管理
                self._init_artists()
                self._init_energy_figure()
                animation.FuncAnimation(
                    self.fig,
                    self._update_artists,
                    frames=frames,
                    interval=interval,
                    blit=False,
                    repeat=False,
                )
                plt.show()
            except Exception as e:
                print(f"无法显示实时动画: {e}")
                print("自动生成能量管理图...")
                plt.switch_backend('Agg')
                self._init_artists()
                self._init_energy_figure()
                self.fig.savefig('simulation.png', dpi=150, bbox_inches='tight')
                self.fig4.savefig('energy.png', dpi=150, bbox_inches='tight')
                print("✓ 图像已保存为 simulation.png, energy.png")


def build_default_simulation() -> Simulator:
    """
    创建默认仿真场景。

    包含1个HAP、3个UAV和20个UE。

    Returns:
        配置好的Simulator实例。
    """
    area_bounds = (0.0, 1000.0, 0.0, 1000.0)  # 仿真区域
    hap = HAP(position=(500.0, 500.0, 600.0))  # HAP位置：高于UAV工作高度的补能平台

    # 创建UAV列表
    uav_list = []
    for uid, position in enumerate([(400.0, 400.0, 200.0), (600.0, 600.0, 220.0), (450.0, 620.0, 180.0)], start=1):
        uav_list.append(
            UAV(
                uid=uid,
                position=np.array(position, dtype=float),
                velocity=np.array([5.0, -3.0, 0.5], dtype=float),
                vmax=20.0,      # 最大速度
                hmin=150.0,     # 最小高度
                hmax=300.0,     # 最大高度
                service_radius=160.0,  # 服务半径
                bounds=area_bounds,    # 边界约束
            )
        )

    # 创建UE列表
    ue_list = []
    rng = np.random.default_rng(seed=42)  # 固定种子以重现
    for uid in range(1, 21):  # 20个UE
        position = np.array(
            [
                rng.uniform(area_bounds[0] + 50.0, area_bounds[1] - 50.0),  # x位置
                rng.uniform(area_bounds[2] + 50.0, area_bounds[3] - 50.0),  # y位置
            ],
            dtype=float,
        )
        ue_list.append(UE(uid=uid, position=position, speed=6.0, bounds=area_bounds))

    a2g_channel = A2GChannel(
        a=9.61,
        b=0.16,
        eta_los=1.6,
        eta_nlos=23.0,
        fc=2.4e9,
        c=3e8,
        bandwidth=10e6,
        transmit_power=0.1,
        noise_power=1e-13,
    )

    a2a_channel = A2AChannel(
        beta0=10.0,      # 大幅增加基准增益系数以提供有效充电
        kappa=1e-3,      # 衰减因子
        bandwidth=10e6,
        transmit_power=0.5,
        noise_power=1e-13,
    )

    # 返回Simulator实例
    return Simulator(
        uav_list=uav_list,
        ue_list=ue_list,
        hap=hap,
        area_bounds=area_bounds,
        a2g_channel=a2g_channel,
        a2a_channel=a2a_channel,
        delta_t=0.5,
    )


if __name__ == "__main__":
    # 主入口：构建默认场景并运行仿真
    import sys

    simulator = build_default_simulation()

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            # 保存为动画文件模式
            save_path = sys.argv[2] if len(sys.argv) > 2 else "simulation.gif"
            print(f"正在生成动画文件: {save_path}")
            simulator.run(frames=200, interval=100, save_path=save_path)
        elif sys.argv[1] == "--image":
            # 保存静态图像模式（仅生成能量图）
            image_path = sys.argv[2] if len(sys.argv) > 2 else "energy.png"
            print(f"正在生成能量管理图像: {image_path}")

            # 运行仿真以收集能量数据
            plt.switch_backend('Agg')
            print("正在运行仿真以收集能量数据...")
            
            simulator.reset()
            for _ in range(300):  # 运行更多帧来更完整地展示返航、充电与恢复
                simulator.step()

            print(f"收集了 {len(simulator.time_history)} 个数据点")

            # 生成能量图
            print("正在生成能量管理图...")
            simulator._init_energy_figure()
            simulator._update_energy_plot()
            simulator.fig4.savefig(image_path, dpi=150, bbox_inches='tight')
            print(f"✓ 能量管理图已保存到: {image_path}")

            # 保存信道性能图和SINR热力图
            simulator._init_performance_figure()
            simulator._update_performance_plot()
            perf_path = image_path.replace('.png', '_performance.png')
            simulator.fig2.savefig(perf_path, dpi=150, bbox_inches='tight')
            print(f"✓ 信道性能图已保存到: {perf_path}")

            simulator._init_heatmap_figure()
            heatmap_path = image_path.replace('.png', '_heatmap.png')
            simulator.fig3.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"✓ SINR热力图已保存到: {heatmap_path}")
            
            plt.close('all')
    else:
        # 实时显示模式
        print("启动实时动画...")
        print("将显示两个窗口：运动仿真 + 实时电量管理")
        print("如果在VS Code中无法显示，请使用以下命令:")
        print("  python env/simulator.py --image energy.png   # 生成能量管理图")
        try:
            simulator.run(frames=700, interval=120)
        except Exception as e:
            print(f"实时显示失败: {e}")
            print("请尝试以下命令生成能量图:")
            print("  python env/simulator.py --image energy.png")
            print("")
            print("正在自动生成能量管理图...")

            # 运行仿真收集数据
            plt.switch_backend('Agg')
            print("正在运行仿真以收集数据...")
            simulator.reset()
            for _ in range(300):  # 运行更多帧来更完整地展示返航、充电与恢复
                simulator.step()

            print(f"收集了 {len(simulator.time_history)} 个数据点")

            # 生成能量图
            print("正在生成能量管理图...")
            simulator._init_energy_figure()
            simulator._update_energy_plot()
            simulator.fig4.savefig('energy.png', dpi=150, bbox_inches='tight')
            print(f"✓ 能量管理图已保存到: energy.png")

            simulator._init_performance_figure()
            simulator._update_performance_plot()
            simulator.fig2.savefig('performance.png', dpi=150, bbox_inches='tight')
            print(f"✓ 信道性能图已保存到: performance.png")

            simulator._init_heatmap_figure()
            simulator.fig3.savefig('heatmap.png', dpi=150, bbox_inches='tight')
            print(f"✓ SINR热力图已保存到: heatmap.png")

            plt.close('all')
