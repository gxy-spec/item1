from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from env.channel import A2AChannel, A2GChannel
from env.energy.energy_model import EnergyModel
from env.mobility.uav import UAV
from env.mobility.ue import UE
from env.simulator import HAP


@dataclass
class AoIEnvConfig:
    area_bounds: Tuple[float, float, float, float] = (0.0, 1000.0, 0.0, 1000.0)
    hap_position: Tuple[float, float, float] = (500.0, 500.0, 600.0)
    uav_position: Tuple[float, float, float] = (400.0, 400.0, 200.0)
    uav_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_t: float = 1.0
    max_steps: int = 200
    num_ues: int = 5
    ue_speed: float = 3.0
    uav_vmax: float = 20.0
    uav_hmin: float = 150.0
    uav_hmax: float = 300.0
    service_radius: float = 180.0
    move_speed: float = 12.0
    vertical_speed: float = 6.0
    packet_size_bits: float = 2e5
    lyapunov_v: float = 2.0
    drift_weight: float = 1.0
    energy_guard_ratio: float = 0.25
    depleted_penalty: float = 2.0
    seed: int = 42


class SingleUAVAoIEnv:
    """单UAV、传统AoI、李雅普诺夫能量队列的一版离散动作环境。"""

    ENERGY_STATES: List[str] = ["normal", "return", "charging", "resume", "depleted"]
    MOVEMENTS: List[str] = ["hover", "north", "south", "east", "west", "up", "down"]

    def __init__(self, config: AoIEnvConfig | None = None):
        self.config = config or AoIEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.current_step = 0

        self.hap = HAP(position=self.config.hap_position)
        self.energy_model = EnergyModel()
        self.a2g_channel = A2GChannel(
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
        self.a2a_channel = A2AChannel(
            beta0=10.0,
            kappa=1e-3,
            bandwidth=10e6,
            transmit_power=0.5,
            noise_power=1e-9,
        )

        self.uav = UAV(
            uid=1,
            position=np.array(self.config.uav_position, dtype=float),
            velocity=np.array(self.config.uav_velocity, dtype=float),
            vmax=self.config.uav_vmax,
            hmin=self.config.uav_hmin,
            hmax=self.config.uav_hmax,
            service_radius=self.config.service_radius,
            bounds=self.config.area_bounds,
        )
        self.uav.energy_max = self.energy_model.E_max
        self.uav.energy = self.energy_model.E_max
        self.uav.home_position = self.uav.position.copy()

        self.ues = self._build_ues()
        self.initial_ue_positions = [ue.position.copy() for ue in self.ues]
        self.initial_uav_position = self.uav.position.copy()
        self.initial_uav_velocity = self.uav.velocity.copy()

        self.aoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.virtual_energy_queue = 0.0

        self.num_actions = 2 * len(self.MOVEMENTS) * (self.config.num_ues + 1)
        self.observation_dim = 8 + len(self.ENERGY_STATES) + 4 * self.config.num_ues

    def _build_ues(self) -> List[UE]:
        xmin, xmax, ymin, ymax = self.config.area_bounds
        ues = []
        for uid in range(1, self.config.num_ues + 1):
            position = np.array(
                [
                    self.rng.uniform(xmin + 80.0, xmax - 80.0),
                    self.rng.uniform(ymin + 80.0, ymax - 80.0),
                ],
                dtype=float,
            )
            ues.append(UE(uid=uid, position=position, speed=self.config.ue_speed, bounds=self.config.area_bounds))
        return ues

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.uav.position = self.initial_uav_position.copy()
        self.uav.velocity = self.initial_uav_velocity.copy()
        self.uav.energy = self.energy_model.E_max
        self.uav.energy_state = "normal"
        self.uav.home_position = self.initial_uav_position.copy()

        for ue, position in zip(self.ues, self.initial_ue_positions):
            ue.position = position.copy()

        self.aoi = np.full(self.config.num_ues, self.config.delta_t, dtype=float)
        self.virtual_energy_queue = 0.0
        return self._get_observation()

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        service_size = self.config.num_ues + 1
        charge_flag = action // (len(self.MOVEMENTS) * service_size)
        remain = action % (len(self.MOVEMENTS) * service_size)
        movement_idx = remain // service_size
        service_idx = remain % service_size
        return charge_flag, movement_idx, service_idx

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        charge_flag, movement_idx, service_idx = self.decode_action(int(action))
        prev_queue = self.virtual_energy_queue

        self._apply_action(charge_flag, movement_idx)
        self._advance_entities()

        selected_ue = None if service_idx == 0 else self.ues[service_idx - 1]
        rate = 0.0
        success = False
        if selected_ue is not None and self.uav.energy_state == "normal":
            rate = self._get_link_rate(selected_ue)
            required_rate = self.config.packet_size_bits / self.config.delta_t
            success = rate >= required_rate

        self._update_aoi(selected_ue, success)
        energy_info = self._update_energy(selected_ue, charge_flag)

        mean_aoi = float(np.mean(self.aoi))
        self.virtual_energy_queue = max(
            prev_queue + self.config.energy_guard_ratio * self.energy_model.E_max - self.uav.energy,
            0.0,
        )
        drift = 0.5 * (self.virtual_energy_queue ** 2 - prev_queue ** 2) / (self.energy_model.E_max ** 2)
        mean_aoi_norm = mean_aoi / (self.config.max_steps * self.config.delta_t)
        reward = -(self.config.lyapunov_v * mean_aoi_norm + self.config.drift_weight * drift)

        if self.uav.energy_state == "depleted":
            reward -= self.config.depleted_penalty

        self.current_step += 1
        done = self.current_step >= self.config.max_steps or self.uav.energy_state == "depleted"

        info = {
            "mean_aoi": mean_aoi,
            "aoi": self.aoi.copy(),
            "selected_ue": None if selected_ue is None else selected_ue.uid,
            "success": success,
            "rate": rate,
            "energy": float(self.uav.energy),
            "energy_state": self.uav.energy_state,
            "virtual_energy_queue": float(self.virtual_energy_queue),
            "drift": float(drift),
            **energy_info,
        }
        return self._get_observation(), float(reward), done, info

    def _apply_action(self, charge_flag: int, movement_idx: int) -> None:
        if self.uav.energy_state in {"charging", "resume", "depleted"}:
            return

        if charge_flag == 1:
            self.uav.energy_state = "return"
            return

        self.uav.energy_state = "normal"
        move_name = self.MOVEMENTS[movement_idx]
        velocity = np.zeros(3, dtype=float)
        if move_name == "north":
            velocity[1] = self.config.move_speed
        elif move_name == "south":
            velocity[1] = -self.config.move_speed
        elif move_name == "east":
            velocity[0] = self.config.move_speed
        elif move_name == "west":
            velocity[0] = -self.config.move_speed
        elif move_name == "up":
            velocity[2] = self.config.vertical_speed
        elif move_name == "down":
            velocity[2] = -self.config.vertical_speed
        self.uav.velocity = velocity

    def _advance_entities(self) -> None:
        self._prepare_uav_motion()
        self.uav.step(self.config.delta_t)
        for ue in self.ues:
            ue.step(self.config.delta_t)

    def _prepare_uav_motion(self) -> None:
        if self.uav.energy_state == "depleted":
            self.uav.velocity[:] = 0.0
            return

        if self.uav.energy_state not in {"return", "resume"}:
            return

        if self.uav.energy_state == "return":
            target = self._get_charging_waypoint()
            arrival_state = "charging"
        else:
            target = self.uav.home_position
            arrival_state = "normal"

        diff = target - self.uav.position
        distance = np.linalg.norm(diff)
        if distance < self.energy_model.charging_distance:
            self.uav.energy_state = arrival_state
            self.uav.velocity[:] = 0.0
            return

        direction = diff / (distance + 1e-6)
        self.uav.velocity = direction * self.uav.vmax

    def _get_charging_waypoint(self) -> np.ndarray:
        charging_height = min(self.uav.hmax, max(self.uav.hmin, self.hap.height - 300.0))
        return np.array([self.hap.xy[0], self.hap.xy[1], charging_height], dtype=float)

    def _get_link_rate(self, selected_ue: UE) -> float:
        results = self.a2g_channel.compute_sinr_and_rate(self.uav, [selected_ue], [self.uav])
        if not results:
            return 0.0
        return float(results[0]["rate"])

    def _update_aoi(self, selected_ue: UE | None, success: bool) -> None:
        self.aoi += self.config.delta_t
        if selected_ue is not None and success:
            self.aoi[selected_ue.uid - 1] = self.config.delta_t

    def _estimate_return_energy_budget(self) -> float:
        target = self._get_charging_waypoint()
        distance = np.linalg.norm(target - self.uav.position)
        travel_time = distance / max(self.uav.vmax, 1e-6)
        cruise_velocity = target - self.uav.position
        norm = np.linalg.norm(cruise_velocity)
        if norm > 1e-6:
            cruise_velocity = cruise_velocity / norm * self.uav.vmax
        else:
            cruise_velocity = np.zeros(3, dtype=float)
        travel_energy = self.energy_model.flying_energy(cruise_velocity, travel_time)
        reserve_energy = 0.05 * self.energy_model.E_max
        return min(self.energy_model.E_max, travel_energy + reserve_energy)

    def _update_energy(self, selected_ue: UE | None, charge_flag: int) -> Dict[str, float]:
        a2a_metrics = self.a2a_channel.compute_link_metrics(self.uav, self.hap)
        channel_gain = a2a_metrics["gain"]

        if self.uav.energy_state == "charging":
            e_fly = 0.0
            e_tx = 0.0
            e_charge = self.energy_model.charging_energy(channel_gain, self.config.delta_t)
            e_charge = max(e_charge, 200.0 * self.config.delta_t)
        elif self.uav.energy_state == "depleted":
            e_fly = 0.0
            e_tx = 0.0
            e_charge = 0.0
        else:
            e_fly = self.energy_model.flying_energy(self.uav.velocity, self.config.delta_t)
            num_covered = 1 if selected_ue is not None and self._is_covered(selected_ue) else 0
            e_tx = self.energy_model.tx_energy(num_covered, self.config.delta_t)
            e_charge = self.energy_model.charging_energy(channel_gain, self.config.delta_t)

        self.uav.energy = self.energy_model.update_battery(self.uav.energy, e_fly, e_tx, e_charge)

        charging_distance = np.linalg.norm(self._get_charging_waypoint() - self.uav.position)
        if self.uav.energy <= 0.0 and self.uav.energy_state not in {"charging", "resume"}:
            if charging_distance < self.energy_model.charging_distance:
                self.uav.energy_state = "charging"
            else:
                self.uav.energy_state = "depleted"
                self.uav.velocity[:] = 0.0
                return {"e_fly": e_fly, "e_tx": e_tx, "e_charge": e_charge}

        dynamic_return_threshold = max(
            self.energy_model.return_threshold * self.energy_model.E_max,
            self._estimate_return_energy_budget(),
        )
        if self.uav.energy <= dynamic_return_threshold and self.uav.energy_state not in {"charging", "depleted", "return"}:
            self.uav.energy_state = "return"

        if self.uav.energy_state == "return" and charging_distance < self.energy_model.charging_distance:
            self.uav.energy_state = "charging"
        elif self.uav.energy_state == "charging":
            self.uav.velocity[:] = 0.0
            if self.energy_model.should_resume_normal(self.uav.energy):
                self.uav.energy_state = "resume"
        elif self.uav.energy_state == "resume":
            if np.linalg.norm(self.uav.home_position - self.uav.position) < self.energy_model.charging_distance:
                self.uav.energy_state = "normal"

        if charge_flag == 1 and self.uav.energy_state == "normal":
            self.uav.energy_state = "return"

        return {"e_fly": e_fly, "e_tx": e_tx, "e_charge": e_charge}

    def _is_covered(self, ue: UE) -> bool:
        dx = self.uav.xy[0] - ue.xy[0]
        dy = self.uav.xy[1] - ue.xy[1]
        return float(np.hypot(dx, dy)) <= self.uav.service_radius

    def _get_observation(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.config.area_bounds
        area_scale_x = max(xmax - xmin, 1.0)
        area_scale_y = max(ymax - ymin, 1.0)

        obs: List[float] = [
            (self.uav.position[0] - xmin) / area_scale_x,
            (self.uav.position[1] - ymin) / area_scale_y,
            (self.uav.position[2] - self.uav.hmin) / max(self.uav.hmax - self.uav.hmin, 1.0),
            self.uav.velocity[0] / max(self.uav.vmax, 1.0),
            self.uav.velocity[1] / max(self.uav.vmax, 1.0),
            self.uav.velocity[2] / max(self.uav.vmax, 1.0),
            self.uav.energy / max(self.energy_model.E_max, 1.0),
            self.virtual_energy_queue / max(self.energy_model.E_max, 1.0),
        ]

        obs.extend([1.0 if self.uav.energy_state == state else 0.0 for state in self.ENERGY_STATES])

        for ue, aoi_value in zip(self.ues, self.aoi):
            obs.append((ue.position[0] - self.uav.position[0]) / area_scale_x)
            obs.append((ue.position[1] - self.uav.position[1]) / area_scale_y)
            obs.append(min(aoi_value / (self.config.max_steps * self.config.delta_t), 1.0))
            obs.append(1.0 if self._is_covered(ue) else 0.0)

        return np.asarray(obs, dtype=np.float32)
