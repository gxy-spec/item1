import os
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np

from channel import A2AChannel, A2GChannel, OFDMAA2GChannel, OFDMAScheduler
from mobility.uav import UAV
from mobility.ue import UE
from simulator import HAP, Simulator


class OFDMASimulator(Simulator):
    """基于 OFDMA A2G 信道的系统级仿真器。"""

    def __init__(
        self,
        uav_list: List[UAV],
        ue_list: List[UE],
        hap: HAP,
        area_bounds: Tuple[float, float, float, float],
        a2g_channel: A2GChannel,
        a2a_channel: A2AChannel,
        ofdma_channel: OFDMAA2GChannel,
        scheduler_mode: str = "aoi_weighted",
        delta_t: float = 0.5,
    ):
        super().__init__(
            uav_list=uav_list,
            ue_list=ue_list,
            hap=hap,
            area_bounds=area_bounds,
            a2g_channel=a2g_channel,
            a2a_channel=a2a_channel,
            delta_t=delta_t,
        )
        self.ofdma_channel = ofdma_channel
        self.scheduler_mode = scheduler_mode
        self.packet_size_bits = 2e5
        self.ue_aoi = {ue.uid: self.delta_t for ue in self.ue_list}
        self.ue_aoi_history = {ue.uid: [] for ue in self.ue_list}

    def reset(self) -> dict:
        observation = super().reset()
        self.ue_aoi = {ue.uid: self.delta_t for ue in self.ue_list}
        self.ue_aoi_history = {ue.uid: [] for ue in self.ue_list}
        return observation

    def _build_assignment(self, uav: UAV) -> Dict[int, List[int]]:
        if self.scheduler_mode == "round_robin":
            return OFDMAScheduler.round_robin(self.ofdma_channel, uav, self.ue_list)
        if self.scheduler_mode == "equal_share":
            return OFDMAScheduler.equal_share(self.ofdma_channel, uav, self.ue_list)
        if self.scheduler_mode == "aoi_weighted":
            weight_map = {ue.uid: float(self.ue_aoi.get(ue.uid, self.delta_t)) for ue in self.ue_list}
            return OFDMAScheduler.aoi_weighted(self.ofdma_channel, uav, self.ue_list, weight_map)
        raise ValueError(f"Unsupported scheduler_mode: {self.scheduler_mode}")

    def _compute_channel_metrics(self, frame: int) -> dict:
        all_sinrs = []
        all_rates = []
        a2a_snrs = []
        uav_sinrs = {}
        uav_rates = {}
        served_uids = set()

        assignments = {uav.uid: self._build_assignment(uav) for uav in self.uav_list}

        for uav in self.uav_list:
            interfering_allocations = [
                (other_uav, assignments[other_uav.uid])
                for other_uav in self.uav_list
                if other_uav.uid != uav.uid
            ]
            metrics = self.ofdma_channel.compute_assignment_metrics(
                serving_uav=uav,
                ue_list=self.ue_list,
                serving_assignment=assignments[uav.uid],
                interfering_allocations=interfering_allocations,
            )

            required_rate = self.packet_size_bits / self.delta_t
            for item in metrics:
                if item["rate"] >= required_rate:
                    served_uids.add(item["ue"].uid)

            ue_sinrs = [item["mean_sinr"] for item in metrics]
            ue_rates = [item["rate"] for item in metrics]
            uav_sinrs[uav.uid] = float(np.mean(ue_sinrs)) if ue_sinrs else float("nan")
            uav_rates[uav.uid] = float(np.mean(ue_rates)) if ue_rates else float("nan")
            all_sinrs.extend(ue_sinrs)
            all_rates.extend(ue_rates)

            a2a_metrics = self.a2a_channel.compute_link_metrics(uav, self.hap)
            a2a_snrs.append(a2a_metrics["snr"])

        avg_a2g_sinr = float(np.mean(all_sinrs)) if all_sinrs else float("nan")
        avg_rate = float(np.mean(all_rates)) if all_rates else float("nan")
        avg_a2a_snr = float(np.mean(a2a_snrs)) if a2a_snrs else float("nan")

        self.time_history.append(frame * self.delta_t)
        self.avg_a2g_sinr_history.append(avg_a2g_sinr)
        self.avg_a2a_snr_history.append(avg_a2a_snr)
        self.avg_rate_history.append(avg_rate)

        for ue in self.ue_list:
            self.ue_aoi[ue.uid] = self.delta_t if ue.uid in served_uids else self.ue_aoi[ue.uid] + self.delta_t
            self.ue_aoi_history[ue.uid].append(self.ue_aoi[ue.uid])

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


def build_default_ofdma_simulation(scheduler_mode: str = "aoi_weighted") -> OFDMASimulator:
    area_bounds = (0.0, 1000.0, 0.0, 1000.0)
    hap = HAP(position=(500.0, 500.0, 600.0))

    uav_list = []
    for uid, position in enumerate([(400.0, 400.0, 200.0), (600.0, 600.0, 220.0), (450.0, 620.0, 180.0)], start=1):
        uav_list.append(
            UAV(
                uid=uid,
                position=np.array(position, dtype=float),
                velocity=np.array([5.0, -3.0, 0.5], dtype=float),
                vmax=20.0,
                hmin=150.0,
                hmax=300.0,
                service_radius=220.0,
                bounds=area_bounds,
            )
        )

    ue_list = []
    rng = np.random.default_rng(seed=42)
    for uid in range(1, 21):
        position = np.array(
            [
                rng.uniform(area_bounds[0] + 50.0, area_bounds[1] - 50.0),
                rng.uniform(area_bounds[2] + 50.0, area_bounds[3] - 50.0),
            ],
            dtype=float,
        )
        ue_list.append(UE(uid=uid, position=position, speed=6.0, bounds=area_bounds))

    # 旧 A2G 只保留给热力图/兼容逻辑使用
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
        beta0=10.0,
        kappa=1e-3,
        bandwidth=10e6,
        transmit_power=0.5,
        noise_power=1e-13,
    )
    ofdma_channel = OFDMAA2GChannel(
        a=9.61,
        b=0.16,
        eta_los=1.6,
        eta_nlos=23.0,
        fc=2.4e9,
        c=3e8,
        total_bandwidth=10e6,
        num_rbs=16,
        total_transmit_power=0.1,
        noise_power_density=1e-20,
    )
    return OFDMASimulator(
        uav_list=uav_list,
        ue_list=ue_list,
        hap=hap,
        area_bounds=area_bounds,
        a2g_channel=a2g_channel,
        a2a_channel=a2a_channel,
        ofdma_channel=ofdma_channel,
        scheduler_mode=scheduler_mode,
        delta_t=0.5,
    )


if __name__ == "__main__":
    simulator = build_default_ofdma_simulation()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            save_path = sys.argv[2] if len(sys.argv) > 2 else "ofdma_output.gif"
            simulator.run(frames=200, interval=100, save_path=save_path)
        elif sys.argv[1] == "--image":
            image_path = sys.argv[2] if len(sys.argv) > 2 else "ofdma_output.png"
            plt.switch_backend("Agg")
            simulator.reset()
            for _ in range(300):
                simulator.step()
            simulator._init_energy_figure()
            simulator._update_energy_plot()
            simulator.fig4.savefig(image_path.replace(".png", "_energy.png"), dpi=150, bbox_inches="tight")
            simulator._init_performance_figure()
            simulator._update_performance_plot()
            simulator.fig2.savefig(image_path.replace(".png", "_performance.png"), dpi=150, bbox_inches="tight")
            simulator.fig.savefig(image_path, dpi=150, bbox_inches="tight")
            plt.close("all")
    else:
        simulator.run(frames=700, interval=120)
