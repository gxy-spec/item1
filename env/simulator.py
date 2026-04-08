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

        # 初始化性能窗口和曲线
        self._init_performance_figure()

        # 初始化热力图窗口
        self._init_heatmap_figure()
        
        # 初始化能量窗口
        self._init_energy_figure()

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
        2. 能量分解堆积图（能耗和充电）
        """
        self.fig4, (self.ax_energy, self.ax_energy_breakdown) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True
        )
        self.fig4.suptitle("UAV Energy Management", fontsize=14)
        
        # ========== 子图1：电量曲线 ==========
        self.ax_energy.set_ylabel("Energy (J)")
        self.ax_energy.set_title("UAV Battery Capacity Over Time")
        self.ax_energy.grid(True, linestyle="--", alpha=0.4)
        
        # 为每个UAV创建电量曲线
        self.energy_lines = {}
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        for i, uav in enumerate(self.uav_list):
            color = colors[i % len(colors)]
            line, = self.ax_energy.plot(
                [], [], c=color, linewidth=2.0,
                label=f"UAV{uav.uid} Energy", marker='o', markersize=3
            )
            self.energy_lines[uav.uid] = line
        
        self.ax_energy.legend(loc="upper right", fontsize=10)
        
        # ========== 子图2：能量分解 ==========
        self.ax_energy_breakdown.set_ylabel("Energy (J)")
        self.ax_energy_breakdown.set_xlabel("Time (s)")
        self.ax_energy_breakdown.set_title("Energy Components (Flying + TX - Charging)")
        self.ax_energy_breakdown.grid(True, linestyle="--", alpha=0.4)
        
        # 能量分解曲线（以第一个UAV为示例）
        if len(self.uav_list) > 0:
            uav0 = self.uav_list[0]
            self.ax_energy_breakdown.plot([], [], c='orange', label='Flying Energy', linewidth=2)
            self.ax_energy_breakdown.plot([], [], c='red', label='TX Energy', linewidth=2)
            self.ax_energy_breakdown.plot([], [], c='green', label='Charging Energy', linewidth=2)
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
        # 更新所有UAV和UE的位置
        for uav in self.uav_list:
            uav.step(self.delta_t)
        for ue in self.ue_list:
            ue.step(self.delta_t)

        # 获取新位置
        uav_positions = np.array([uav.xy for uav in self.uav_list])
        ue_positions = np.array([ue.xy for ue in self.ue_list])

        # 更新散点图位置
        self.uav_scatter.set_offsets(uav_positions)
        self.ue_scatter.set_offsets(ue_positions)

        # 更新服务圆圈位置
        for circle, uav in zip(self.service_circles, self.uav_list):
            circle.center = tuple(uav.xy)

        # 计算信道性能指标
        metrics = self._compute_channel_metrics(frame)

        # 更新高度显示文本与当前性能指标
        # 显示格式更清晰
        time_str = f"Time: {frame * self.delta_t:.1f}s"
        altitudes = " | ".join([f"UAV{uav.uid}: {uav.height:.0f}m" for uav in self.uav_list])
        uav_sinr_text = " | ".join([f"SINR{uav.uid}: {metrics['uav_sinrs'][uav.uid]:.2f}" for uav in self.uav_list])
        uav_rate_text = " | ".join([f"R{uav.uid}: {metrics['uav_rates'][uav.uid] / 1e6:.1f}M" for uav in self.uav_list])
        self.altitude_text.set_text(
            f"{time_str}  |  HAP: {self.hap.height:.0f}m\nAlt: {altitudes}\n{uav_sinr_text}\n{uav_rate_text} Mbps"
        )

        self._update_performance_plot()
        
        # 更新能量窗口
        self._update_energy_plot()

        # 更新热力图
        self._update_heatmap()
        
        # ============ 计算和更新能量 ============
        self._update_energy(metrics)

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
            a2g_results = self.a2g_channel.compute_sinr_and_rate(uav, self.ue_list)
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
            # ========== 步骤1：计算飞行能耗 ==========
            E_fly = self.energy_model.flying_energy(uav.velocity, self.delta_t)
            
            # ========== 步骤2：计算通信能耗 ==========
            covered_ues = self.a2g_channel.find_covered_ues(uav, self.ue_list)
            num_covered = len(covered_ues)
            E_tx = self.energy_model.tx_energy(num_covered, self.delta_t)
            
            # ========== 步骤3-4：计算充电能量 ==========
            # 从A2A信道获取UAV到HAP的链路增益
            a2a_metrics = self.a2a_channel.compute_link_metrics(uav, self.hap)
            channel_gain_nH = a2a_metrics["gain"]  # 幅度（线性度）
            E_charge = self.energy_model.charging_energy(channel_gain_nH, self.delta_t)
            
            # ========== 步骤5：更新电池容量 ==========
            uav.energy = self.energy_model.update_battery(
                uav.energy,
                E_fly,
                E_tx,
                E_charge,
            )
            
            # ========== 步骤6a：状态机逻辑 - 判断是否需要返回充电 ==========
            if self.energy_model.should_return_to_charging(uav.energy):
                if uav.energy_state != "charging":
                    uav.set_energy_state("return")
            
            # ========== 步骤6b：处理"return"状态 - 飞向HAP ==========
            if uav.energy_state == "return":
                # 计算UAV到HAP的距离
                diff = self.hap.xy - uav.xy
                distance = np.linalg.norm(diff)
                
                # 如果已接近HAP，进入充电状态
                if distance < self.energy_model.charging_distance:
                    uav.set_energy_state("charging")
                else:
                    # 飞向HAP：修改速度指向HAP
                    direction = diff / (distance + 1e-6)
                    speed = np.linalg.norm(uav.velocity)
                    uav.velocity[:2] = direction * min(speed, uav.vmax)
            
            # ========== 步骤6c：处理"charging"状态 - 停止运动 ==========
            elif uav.energy_state == "charging":
                # 停止水平运动，垂直速度为0
                uav.velocity[:2] = 0.0
                uav.velocity[2] = 0.0
                
                # 检查是否充满电，可以恢复正常
                if self.energy_model.should_resume_normal(uav.energy):
                    uav.set_energy_state("normal")
            
            # ========== 步骤6d：处理"normal"状态 - 正常随机运动 ==========
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
                # 使用最少的数据点数（防止长度不匹配）
                min_len = min(len(x), len(self.energy_history[uav.uid]))
                self.energy_lines[uav.uid].set_data(x[:min_len], np.array(self.energy_history[uav.uid][:min_len]))
        
        # 重新计算自动缩放
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
        # ========== 更新能量分解 ==========
        if len(self.uav_list) > 0 and len(x) > 0:
            uav0 = self.uav_list[0]
            flying = np.array(self.flying_energy_history.get(uav0.uid, []))
            tx = np.array(self.tx_energy_history.get(uav0.uid, []))
            charging = np.array(self.charging_energy_history.get(uav0.uid, []))
            
            # 更新分解图的曲线，只在有数据时更新
            if len(flying) > 0 and len(tx) > 0 and len(charging) > 0:
                min_len = min(len(x), len(flying), len(tx), len(charging))
                lines = self.ax_energy_breakdown.get_lines()
                if len(lines) >= 3:
                    lines[0].set_data(x[:min_len], flying[:min_len])
                    lines[1].set_data(x[:min_len], tx[:min_len])
                    lines[2].set_data(x[:min_len], charging[:min_len])
            
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
        if save_path is not None:
            # 如果要保存为文件，使用非GUI后端
            plt.switch_backend('Agg')

            # 先运行仿真找到最佳热力图时刻
            print("正在分析仿真以找到最佳热力图时刻...")
            best_frame = 0
            max_ue_count = 0
            frame_positions = []

            for frame in range(frames):
                # 保存当前位置
                uav_positions = [uav.position.copy() for uav in self.uav_list]
                ue_positions = [ue.position.copy() for ue in self.ue_list]
                frame_positions.append((uav_positions, ue_positions))

                # 更新位置
                for uav in self.uav_list:
                    uav.step(self.delta_t)
                for ue in self.ue_list:
                    ue.step(self.delta_t)

                # 计算并记录性能指标
                metrics = self._compute_channel_metrics(frame)

                # 检查第一个UAV覆盖范围内的UE数量
                covered_count = len(self.a2g_channel.find_covered_ues(self.uav_list[0], self.ue_list))
                if covered_count > max_ue_count:
                    max_ue_count = covered_count
                    best_frame = frame

            print(f"找到最佳热力图时刻: 帧{best_frame}, UAV覆盖范围内有{max_ue_count}个UE")

            # 回放到最佳时刻
            if best_frame < len(frame_positions):
                uav_positions, ue_positions = frame_positions[best_frame]
                for uav, pos in zip(self.uav_list, uav_positions):
                    uav.position = pos
                for ue, pos in zip(self.ue_list, ue_positions):
                    ue.position = pos

            # 初始化所有窗口（在最佳时刻）
            self._init_artists()
            self._init_performance_figure()
            self._init_heatmap_figure()

            # 创建主窗口动画（重新运行仿真）
            # 重置位置到开始状态
            for frame_idx, (uav_pos, ue_pos) in enumerate(frame_positions):
                for uav, pos in zip(self.uav_list, uav_pos):
                    uav.position = pos
                for ue, pos in zip(self.ue_list, ue_pos):
                    ue.position = pos
                break  # 只重置到第一帧

            anim = animation.FuncAnimation(
                self.fig,
                self._update_artists,
                frames=frames,
                interval=interval,
                blit=False,
                repeat=False,
            )

            # 保存主窗口动画
            try:
                writer = animation.PillowWriter(fps=1000/interval)
                anim.save(save_path, writer=writer)
                print(f"主窗口动画已保存到: {save_path}")
            except Exception as e:
                print(f"保存动画失败: {e}")
                print("请确保安装了必要的依赖: pip install pillow")

            # 保存性能曲线和热力图的静态图像（在最佳时刻）
            try:
                # 性能曲线显示完整趋势
                perf_path = save_path.replace('.gif', '_performance.png')
                self.fig2.savefig(perf_path, dpi=150, bbox_inches='tight')
                print(f"性能曲线图已保存到: {perf_path}")

                # 热力图显示最佳时刻
                heatmap_path = save_path.replace('.gif', '_heatmap.png')
                self.fig3.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                print(f"热力图已保存到: {heatmap_path}")

            except Exception as e:
                print(f"保存静态图像失败: {e}")

        else:
            # 实时显示模式
            try:
                # 尝试使用交互式后端
                plt.switch_backend('TkAgg')
                # 初始化所有窗口
                self._init_artists()
                self._init_performance_figure()
                self._init_heatmap_figure()
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
                print("自动生成静态图像...")
                plt.switch_backend('Agg')
                self._init_artists()
                self._init_performance_figure()
                self._init_heatmap_figure()
                self.fig.savefig('simulation.png', dpi=150, bbox_inches='tight')
                self.fig2.savefig('performance.png', dpi=150, bbox_inches='tight')
                self.fig3.savefig('heatmap.png', dpi=150, bbox_inches='tight')
                print("静态图像已保存为 simulation.png, performance.png, heatmap.png")


def build_default_simulation() -> Simulator:
    """
    创建默认仿真场景。

    包含1个HAP、3个UAV和20个UE。

    Returns:
        配置好的Simulator实例。
    """
    area_bounds = (0.0, 1000.0, 0.0, 1000.0)  # 仿真区域
    hap = HAP(position=(500.0, 500.0, 2000.0))  # HAP位置

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
        noise_power=1e-9,
    )

    a2a_channel = A2AChannel(
        beta0=1e-3,
        kappa=1e-4,
        bandwidth=10e6,
        transmit_power=0.5,
        noise_power=1e-9,
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
            simulator.run(frames=100, interval=100, save_path=save_path)
        elif sys.argv[1] == "--image":
            # 保存静态图像模式
            image_path = sys.argv[2] if len(sys.argv) > 2 else "simulation.png"
            print(f"正在生成静态图像: {image_path}")

            # 运行仿真找到最佳热力图时刻
            plt.switch_backend('Agg')
            print("正在分析仿真以找到最佳热力图时刻...")
            best_frame = 0
            max_ue_count = 0
            frame_positions = []

            for frame in range(200):  # 运行200帧来收集数据
                # 保存当前位置
                uav_positions = [simulator.uav_list[i].position.copy() for i in range(len(simulator.uav_list))]
                ue_positions = [simulator.ue_list[i].position.copy() for i in range(len(simulator.ue_list))]
                frame_positions.append((uav_positions, ue_positions))

                # 更新位置
                for uav in simulator.uav_list:
                    uav.step(simulator.delta_t)
                for ue in simulator.ue_list:
                    ue.step(simulator.delta_t)

                # 计算并记录性能指标
                metrics = simulator._compute_channel_metrics(frame)
                
                # 计算并更新能量
                simulator._update_energy(metrics)

                # 检查第一个UAV覆盖范围内的UE数量
                covered_count = len(simulator.a2g_channel.find_covered_ues(simulator.uav_list[0], simulator.ue_list))
                if covered_count > max_ue_count:
                    max_ue_count = covered_count
                    best_frame = frame

            print(f"找到最佳热力图时刻: 帧{best_frame}, UAV覆盖范围内有{max_ue_count}个UE")
            print(f"收集了 {len(simulator.time_history)} 个数据点")

            # 生成可视化
            print("正在生成可视化...")

            # 主窗口（使用第一帧）
            if len(frame_positions) > 0:
                uav_positions, ue_positions = frame_positions[0]
                for uav, pos in zip(simulator.uav_list, uav_positions):
                    uav.position = pos
                for ue, pos in zip(simulator.ue_list, ue_positions):
                    ue.position = pos

            simulator._init_artists()
            simulator.fig.savefig(image_path, dpi=150, bbox_inches='tight')
            print(f"运动仿真图已保存到: {image_path}")

            # 性能曲线图（显示完整趋势）
            perf_path = image_path.replace('.png', '_performance.png')
            fig_perf, (ax_sinr, ax_snr, ax_rate) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
            fig_perf.suptitle("Channel Performance Metrics", fontsize=14)

            if len(simulator.time_history) > 0:
                x = np.array(simulator.time_history)
                y_sinr = np.array(simulator.avg_a2g_sinr_history)
                y_snr = np.array(simulator.avg_a2a_snr_history)
                y_rate = np.array(simulator.avg_rate_history)

                # 绘制平均值曲线
                ax_sinr.plot(x, y_sinr, c="tab:blue", label="Avg A2G SINR", linewidth=2, linestyle='--')
                
                # 绘制每个UAV的SINR曲线
                colors = ['tab:red', 'tab:purple', 'tab:brown']
                for i, uav in enumerate(simulator.uav_list):
                    if uav.uid in simulator.uav_sinr_history:
                        y_uav_sinr = np.array(simulator.uav_sinr_history[uav.uid])
                        ax_sinr.plot(x, y_uav_sinr, c=colors[i % len(colors)], 
                                   label=f"UAV{uav.uid} SINR", linewidth=1.5, alpha=0.8)
                
                ax_sinr.set_ylabel("A2G SINR (dB)")
                ax_sinr.grid(True, linestyle="--", alpha=0.4)
                ax_sinr.legend(loc="upper right")

                ax_snr.plot(x, y_snr, c="tab:orange", label="Avg A2A SNR", linewidth=2)
                ax_snr.set_ylabel("A2A SNR (dB)")
                ax_snr.grid(True, linestyle="--", alpha=0.4)
                ax_snr.legend(loc="upper right")

                # 绘制平均速率曲线
                ax_rate.plot(x, y_rate, c="tab:green", label="Avg Rate", linewidth=2, linestyle='--')
                
                # 绘制每个UAV的速率曲线
                for i, uav in enumerate(simulator.uav_list):
                    if uav.uid in simulator.uav_rate_history:
                        y_uav_rate = np.array(simulator.uav_rate_history[uav.uid])
                        ax_rate.plot(x, y_uav_rate, c=colors[i % len(colors)], 
                                   label=f"UAV{uav.uid} Rate", linewidth=1.5, alpha=0.8)
                
                ax_rate.set_ylabel("Rate (bps)")
                ax_rate.set_xlabel("Time (s)")
                ax_rate.grid(True, linestyle="--", alpha=0.4)
                ax_rate.legend(loc="upper right")

            fig_perf.tight_layout()
            fig_perf.savefig(perf_path, dpi=150, bbox_inches='tight')
            print(f"性能曲线图已保存到: {perf_path}")

            # 热力图（使用最佳时刻）
            if best_frame < len(frame_positions):
                uav_positions, ue_positions = frame_positions[best_frame]
                for uav, pos in zip(simulator.uav_list, uav_positions):
                    uav.position = pos
                for ue, pos in zip(simulator.ue_list, ue_positions):
                    ue.position = pos

            heatmap_path = image_path.replace('.png', '_heatmap.png')
            simulator._init_heatmap_figure()
            simulator._update_heatmap()
            simulator.fig3.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"热力图已保存到: {heatmap_path}")

            # 能量图（显示各UAV的电量和能耗）
            energy_path = image_path.replace('.png', '_energy.png')
            simulator._init_energy_figure()
            simulator._update_energy_plot()
            simulator.fig4.savefig(energy_path, dpi=150, bbox_inches='tight')
            print(f"能量管理图已保存到: {energy_path}")

            plt.close('all')
    else:
        # 实时显示模式
        print("启动实时动画...")
        print("将显示三个窗口：运动仿真、性能曲线、SINR热力图")
        print("如果在VS Code中无法显示，请使用以下命令之一:")
        print("  python env/simulator.py --save simulation.gif    # 保存为GIF动画")
        print("  python env/simulator.py --image simulation.png   # 保存静态图像")
        try:
            simulator.run(frames=500, interval=120)
        except Exception as e:
            print(f"实时显示失败: {e}")
            print("建议使用以下命令生成高质量图像:")
            print("  python env/simulator.py --save simulation.gif    # 保存为动画GIF")
            print("  python env/simulator.py --image simulation.png   # 保存静态图像")
            print("")
            print("正在自动生成静态图像...")

            # 运行仿真收集数据
            plt.switch_backend('Agg')
            print("正在运行仿真以收集数据...")
            for frame in range(200):  # 运行200帧来收集数据
                for uav in simulator.uav_list:
                    uav.step(simulator.delta_t)
                for ue in simulator.ue_list:
                    ue.step(simulator.delta_t)
                metrics = simulator._compute_channel_metrics(frame)
                simulator._update_energy(metrics)

            print(f"收集了 {len(simulator.time_history)} 个数据点")

            # 生成静态图像
            print("正在生成可视化...")

            # 创建运动仿真图
            simulator._init_artists()
            simulator.fig.savefig('simulation.png', dpi=150, bbox_inches='tight')

            # 创建性能曲线图
            fig_perf, (ax_sinr, ax_snr, ax_rate) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
            fig_perf.suptitle("Channel Performance Metrics", fontsize=14)

            if len(simulator.time_history) > 0:
                x = np.array(simulator.time_history)
                y_sinr = np.array(simulator.avg_a2g_sinr_history)
                y_snr = np.array(simulator.avg_a2a_snr_history)
                y_rate = np.array(simulator.avg_rate_history)

                ax_sinr.plot(x, y_sinr, c="tab:blue", label="Avg A2G SINR", linewidth=2)
                ax_sinr.set_ylabel("Avg A2G SINR")
                ax_sinr.grid(True, linestyle="--", alpha=0.4)
                ax_sinr.legend(loc="upper right")

                ax_snr.plot(x, y_snr, c="tab:orange", label="Avg A2A SNR", linewidth=2)
                ax_snr.set_ylabel("Avg A2A SNR")
                ax_snr.grid(True, linestyle="--", alpha=0.4)
                ax_snr.legend(loc="upper right")

                ax_rate.plot(x, y_rate, c="tab:green", label="Avg Rate", linewidth=2)
                ax_rate.set_ylabel("Avg Rate (bps)")
                ax_rate.set_xlabel("Time (s)")
                ax_rate.grid(True, linestyle="--", alpha=0.4)
                ax_rate.legend(loc="upper right")

            fig_perf.tight_layout()
            fig_perf.savefig('performance.png', dpi=150, bbox_inches='tight')

            # 创建热力图
            simulator._init_heatmap_figure()
            simulator._update_heatmap()
            simulator.fig3.savefig('heatmap.png', dpi=150, bbox_inches='tight')

            # 创建能量图
            simulator._init_energy_figure()
            simulator._update_energy_plot()
            simulator.fig4.savefig('energy.png', dpi=150, bbox_inches='tight')

            plt.close('all')  # 关闭所有图形
            print("静态图像已保存为 simulation.png, performance.png, heatmap.png, energy.png")
