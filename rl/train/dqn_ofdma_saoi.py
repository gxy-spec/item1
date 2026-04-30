from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.agents import QNetwork
from rl.analysis import plot_training_curves
from rl.envs import OFDMASAoIEnvConfig, SingleUAVOFDMASAoIEnv


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


def ensure_unique_path(output_dir: str | Path, stem: str, suffix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    env = SingleUAVOFDMASAoIEnv(OFDMASAoIEnvConfig(max_steps=args.max_steps, num_ues=args.num_ues, seed=args.seed))
    csv_path = build_unique_csv_path(args.output_dir, args.csv_name)
    best_model_path = ensure_unique_path(args.model_output_dir, "dqn_ofdma_saoi_best", ".pth")

    q_net = QNetwork(env.observation_dim, env.num_actions).to(device)
    target_net = QNetwork(env.observation_dim, env.num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = deque(maxlen=args.buffer_size)

    epsilon = args.epsilon_start
    global_step = 0
    reward_window = deque(maxlen=20)
    saoi_window = deque(maxlen=20)
    best_score = -float("inf")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode", "reward", "reward_ma20", "avg_saoi", "avg_saoi_ma20", "final_saoi",
                "avg_aoi", "success_updates", "service_attempts", "success_rate", "charge_steps",
                "final_energy", "min_energy", "queue", "epsilon", "best_model_path"
            ],
        )
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            obs = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            done = False
            episode_mean_saoi = []
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
                episode_mean_saoi.append(info["mean_saoi"])
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
            avg_mean_saoi = float(np.mean(episode_mean_saoi)) if episode_mean_saoi else 0.0
            success_rate = success_updates / max(service_attempts, 1)
            reward_window.append(episode_reward)
            saoi_window.append(avg_mean_saoi)
            row = {
                "episode": episode,
                "reward": episode_reward,
                "reward_ma20": float(np.mean(reward_window)),
                "avg_saoi": avg_mean_saoi,
                "avg_saoi_ma20": float(np.mean(saoi_window)),
                "final_saoi": float(info["mean_saoi"]),
                "avg_aoi": float(info["mean_aoi"]),
                "success_updates": success_updates,
                "service_attempts": service_attempts,
                "success_rate": success_rate,
                "charge_steps": charge_steps,
                "final_energy": float(info["energy"]),
                "min_energy": float(min_energy),
                "queue": float(info["virtual_energy_queue"]),
                "epsilon": float(epsilon),
                "best_model_path": "",
            }
            save_score = row["reward_ma20"] - 0.25 * row["avg_saoi_ma20"]
            if save_score >= best_score:
                best_score = save_score
                torch.save({"model_state": q_net.state_dict()}, best_model_path)
                row["best_model_path"] = str(best_model_path)
                print(f"saved_model(best_dqn_ofdma_saoi_checkpoint)={best_model_path}")
            writer.writerow(row)
            csv_file.flush()

            print(
                f"ep={episode} "
                f"reward={episode_reward:.3f} "
                f"reward_ma20={row['reward_ma20']:.3f} "
                f"avg_saoi={avg_mean_saoi:.3f} "
                f"success_rate={success_updates}/{service_attempts}={success_rate:.2f} "
                f"charge_steps={charge_steps} "
                f"final_energy={info['energy']:.1f} "
                f"min_energy={min_energy:.1f} "
                f"queue={info['virtual_energy_queue']:.1f} "
                f"epsilon={epsilon:.3f}"
            )

    print(f"saved_csv(ofdma_saoi_training_metrics)={csv_path}")
    print(f"saved_model(dqn_ofdma_saoi_checkpoint)={best_model_path}")
    if args.auto_plot:
        plot_path = plot_training_curves(csv_path=csv_path, output_dir=args.plot_output_dir)
        print(f"saved_plot(training_curves)={plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on OFDMA SAoI environment")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/training/logs")
    parser.add_argument("--csv-name", type=str, default="ofdma_saoi_training_metrics.csv")
    parser.add_argument("--plot-output-dir", type=str, default="rl/outputs/training/plots")
    parser.add_argument("--model-output-dir", type=str, default="rl/outputs/training/models")
    parser.add_argument("--auto-plot", dest="auto_plot", action="store_true")
    parser.add_argument("--no-auto-plot", dest="auto_plot", action="store_false")
    parser.set_defaults(auto_plot=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
