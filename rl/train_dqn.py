from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.aoi_energy_env import AoIEnvConfig, SingleUAVAoIEnv
from rl.plot_training_curves import plot_training_curves


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


def build_unique_csv_path(output_dir: str | Path, csv_name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(csv_name).stem
    suffix = Path(csv_name).suffix or ".csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{stem}_{timestamp}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{stem}_{timestamp}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SingleUAVAoIEnv(AoIEnvConfig(max_steps=args.max_steps, num_ues=args.num_ues, seed=args.seed))
    csv_path = build_unique_csv_path(args.output_dir, args.csv_name)

    q_net = QNetwork(env.observation_dim, env.num_actions).to(device)
    target_net = QNetwork(env.observation_dim, env.num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = deque(maxlen=args.buffer_size)

    epsilon = args.epsilon_start
    global_step = 0
    reward_window = deque(maxlen=20)
    aoi_window = deque(maxlen=20)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "reward",
                "reward_ma20",
                "avg_aoi",
                "avg_aoi_ma20",
                "final_aoi",
                "success_updates",
                "service_attempts",
                "success_rate",
                "charge_steps",
                "final_energy",
                "min_energy",
                "queue",
                "epsilon",
            ],
        )
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            obs = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            done = False
            episode_mean_aois = []
            success_updates = 0
            service_attempts = 0
            charge_steps = 0
            min_energy = env.uav.energy

            while not done:
                global_step += 1
                if random.random() < epsilon:
                    action = random.randrange(env.num_actions)
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(torch.argmax(q_net(obs_tensor), dim=1).item())

                next_obs, reward, done, info = env.step(action)
                replay.append(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
                obs = next_obs
                episode_reward += reward
                episode_mean_aois.append(info["mean_aoi"])
                min_energy = min(min_energy, info["energy"])
                if info["selected_ue"] is not None:
                    service_attempts += 1
                if info["success"]:
                    success_updates += 1
                if info["energy_state"] in {"return", "charging", "resume"}:
                    charge_steps += 1

                if len(replay) >= args.batch_size:
                    batch = random.sample(replay, args.batch_size)
                    obs_batch = torch.tensor(np.asarray([item.obs for item in batch]), dtype=torch.float32, device=device)
                    action_batch = torch.tensor([item.action for item in batch], dtype=torch.int64, device=device).unsqueeze(1)
                    reward_batch = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
                    next_obs_batch = torch.tensor(np.asarray([item.next_obs for item in batch]), dtype=torch.float32, device=device)
                    done_batch = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)

                    q_values = q_net(obs_batch).gather(1, action_batch).squeeze(1)
                    with torch.no_grad():
                        next_q = target_net(next_obs_batch).max(dim=1).values
                        target = reward_batch + args.gamma * (1.0 - done_batch) * next_q

                    loss = nn.functional.mse_loss(q_values, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % args.target_update == 0:
                    target_net.load_state_dict(q_net.state_dict())

            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
            avg_mean_aoi = float(np.mean(episode_mean_aois)) if episode_mean_aois else 0.0
            success_rate = success_updates / max(service_attempts, 1)
            reward_window.append(episode_reward)
            aoi_window.append(avg_mean_aoi)
            row = {
                "episode": episode,
                "reward": episode_reward,
                "reward_ma20": float(np.mean(reward_window)),
                "avg_aoi": avg_mean_aoi,
                "avg_aoi_ma20": float(np.mean(aoi_window)),
                "final_aoi": float(info["mean_aoi"]),
                "success_updates": success_updates,
                "service_attempts": service_attempts,
                "success_rate": success_rate,
                "charge_steps": charge_steps,
                "final_energy": float(info["energy"]),
                "min_energy": float(min_energy),
                "queue": float(info["virtual_energy_queue"]),
                "epsilon": float(epsilon),
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"ep={episode} "
                f"reward(回合奖励)={episode_reward:.3f} "
                f"reward_ma20(20轮平均奖励)={row['reward_ma20']:.3f} "
                f"avg_aoi(平均信息年龄)={avg_mean_aoi:.3f} "
                f"final_aoi(最终信息年龄)={info['mean_aoi']:.3f} "
                f"success_rate(更新成功率)={success_updates}/{service_attempts}={success_rate:.2f} "
                f"charge_steps(充电相关步数)={charge_steps} "
                f"final_energy(最终能量)={info['energy']:.1f} "
                f"min_energy(最低能量)={min_energy:.1f} "
                f"queue(虚拟能量队列)={info['virtual_energy_queue']:.1f} "
                f"epsilon(探索率)={epsilon:.3f}"
            )

    print(f"saved_csv(training_metrics)={csv_path}")

    if args.auto_plot:
        plot_path = plot_training_curves(csv_path=csv_path, output_dir=args.plot_output_dir)
        print(f"saved_plot(training_curves)={plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple DQN on the AoI-energy environment")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/results")
    parser.add_argument("--csv-name", type=str, default="training_metrics.csv")
    parser.add_argument("--plot-output-dir", type=str, default="rl/plots")
    parser.add_argument("--auto-plot", dest="auto_plot", action="store_true")
    parser.add_argument("--no-auto-plot", dest="auto_plot", action="store_false")
    parser.set_defaults(auto_plot=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
