from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rl.agents import SACAgent
from rl.analysis import plot_training_curves
from rl.baselines.continuous_semantic_env import continuous_rule_semantic_policy
from rl.envs import ContinuousSemanticSAoIEnv, ContinuousSemanticSAoIEnvConfig


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: list[Transition] = []
        self.position = 0

    def add(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[idx] for idx in indices]
        return {
            "obs": torch.as_tensor(np.asarray([item.obs for item in batch]), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.asarray([item.action for item in batch]), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.asarray([item.reward for item in batch]), dtype=torch.float32, device=device).unsqueeze(1),
            "next_obs": torch.as_tensor(np.asarray([item.next_obs for item in batch]), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.asarray([item.done for item in batch]), dtype=torch.float32, device=device).unsqueeze(1),
        }

    def __len__(self) -> int:
        return len(self.storage)


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


def heuristic_warmup_action(
    env: ContinuousSemanticSAoIEnv,
    obs: np.ndarray,
    noise_std_xy: float,
    noise_std_charge: float,
    noise_std_service: float,
    noise_std_ratio: float,
) -> np.ndarray:
    base_action = continuous_rule_semantic_policy(env, obs).astype(np.float32)
    noise = np.array(
        [
            env.rng.normal(0.0, noise_std_xy),
            env.rng.normal(0.0, noise_std_xy),
            env.rng.normal(0.0, noise_std_charge),
            env.rng.normal(0.0, noise_std_service),
            env.rng.normal(0.0, noise_std_ratio),
        ],
        dtype=np.float32,
    )
    action = base_action + noise
    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1] = float(np.clip(action[1], -1.0, 1.0))
    action[2] = float(np.clip(action[2], 0.0, 1.0))
    action[3] = float(np.clip(action[3], 0.0, 1.0))
    action[4] = float(np.clip(action[4], 0.0, 1.0))
    return action


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ContinuousSemanticSAoIEnv(
        ContinuousSemanticSAoIEnvConfig(max_steps=args.max_steps, num_ues=args.num_ues, seed=args.seed)
    )
    agent = SACAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
    )

    replay = ReplayBuffer(args.buffer_size)
    csv_path = ensure_unique_path(args.output_dir, "semantic_sac_training_metrics", ".csv")
    best_model_path = ensure_unique_path(args.model_output_dir, "semantic_sac_best", ".pth")
    reward_window = deque(maxlen=20)
    saoi_window = deque(maxlen=20)
    aoi_window = deque(maxlen=20)
    global_step = 0
    best_reward_ma20 = -float("inf")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "reward",
                "reward_ma20",
                "avg_saoi",
                "avg_saoi_ma20",
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
                "final_queue",
                "max_queue",
                "alpha",
                "actor_loss",
                "critic_loss",
                "best_model_path",
            ],
        )
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            obs = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            mean_saois = []
            mean_aois = []
            success_updates = 0
            service_attempts = 0
            charge_steps = 0
            min_energy = env.uav.energy
            queue_values = []
            actor_losses = []
            critic_losses = []
            alpha_values = []
            done = False

            while not done:
                global_step += 1
                if global_step <= args.start_steps:
                    action = heuristic_warmup_action(
                        env=env,
                        obs=obs,
                        noise_std_xy=args.warmup_noise_xy,
                        noise_std_charge=args.warmup_noise_charge,
                        noise_std_service=args.warmup_noise_service,
                        noise_std_ratio=args.warmup_noise_ratio,
                    )
                elif episode <= args.guided_episodes and env.rng.random() < args.guided_exploration_prob:
                    action = heuristic_warmup_action(
                        env=env,
                        obs=obs,
                        noise_std_xy=args.guided_noise_xy,
                        noise_std_charge=args.guided_noise_charge,
                        noise_std_service=args.guided_noise_service,
                        noise_std_ratio=args.guided_noise_ratio,
                    )
                else:
                    action = agent.select_action(obs, evaluate=False).astype(np.float32)

                next_obs, reward, done, info = env.step(action)
                replay.add(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
                obs = next_obs
                episode_reward += reward
                mean_saois.append(info["mean_saoi"])
                mean_aois.append(info["mean_aoi"])
                min_energy = min(min_energy, float(info["energy"]))
                if info["selected_ue"] is not None:
                    service_attempts += 1
                if info["success"]:
                    success_updates += 1
                if info["energy_state"] in {"return", "charging", "resume"}:
                    charge_steps += 1
                queue_values.append(float(info["virtual_energy_queue"]))

                if len(replay) >= args.batch_size and global_step >= args.update_after:
                    for _ in range(args.updates_per_step):
                        batch = replay.sample(args.batch_size, device=device)
                        stats = agent.update(batch)
                        actor_losses.append(stats.actor_loss)
                        critic_losses.append(stats.critic_loss)
                        alpha_values.append(stats.alpha)

            avg_mean_saoi = float(np.mean(mean_saois)) if mean_saois else 0.0
            avg_mean_aoi = float(np.mean(mean_aois)) if mean_aois else 0.0
            success_rate = success_updates / max(service_attempts, 1)
            reward_window.append(episode_reward)
            saoi_window.append(avg_mean_saoi)
            aoi_window.append(avg_mean_aoi)
            reward_ma20 = float(np.mean(reward_window))
            avg_queue = float(np.mean(queue_values)) if queue_values else 0.0
            final_queue = float(queue_values[-1]) if queue_values else 0.0
            max_queue = float(np.max(queue_values)) if queue_values else 0.0
            saved_best = False

            row = {
                "episode": episode,
                "reward": episode_reward,
                "reward_ma20": reward_ma20,
                "avg_saoi": avg_mean_saoi,
                "avg_saoi_ma20": float(np.mean(saoi_window)),
                "avg_aoi": avg_mean_aoi,
                "avg_aoi_ma20": float(np.mean(aoi_window)),
                "final_aoi": float(info["mean_aoi"]),
                "success_updates": success_updates,
                "service_attempts": service_attempts,
                "success_rate": success_rate,
                "charge_steps": charge_steps,
                "final_energy": float(info["energy"]),
                "min_energy": float(min_energy),
                "queue": avg_queue,
                "final_queue": final_queue,
                "max_queue": max_queue,
                "alpha": float(np.mean(alpha_values)) if alpha_values else float(agent.alpha.item()),
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                "best_model_path": "",
            }

            if reward_ma20 >= best_reward_ma20:
                best_reward_ma20 = reward_ma20
                agent.save(best_model_path)
                row["best_model_path"] = str(best_model_path)
                saved_best = True

            writer.writerow(row)
            csv_file.flush()

            print(
                f"ep={episode} reward={episode_reward:.3f} reward_ma20={reward_ma20:.3f} "
                f"avg_saoi={avg_mean_saoi:.3f} avg_aoi={avg_mean_aoi:.3f} "
                f"success_rate={success_updates}/{service_attempts}={success_rate:.2f} "
                f"charge_steps={charge_steps} final_energy={info['energy']:.1f} avg_queue={avg_queue:.1f}"
            )
            if saved_best:
                print(f"saved_model(best_semantic_sac_checkpoint)={best_model_path}")

    print(f"saved_csv(semantic_sac_training_metrics)={csv_path}")
    print(f"saved_model(semantic_sac_checkpoint)={best_model_path}")

    if args.auto_plot:
        plot_path = plot_training_curves(csv_path=csv_path, output_dir=args.plot_output_dir)
        print(f"saved_plot(training_curves)={plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a semantic SAC baseline with user association and compression control")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--start-steps", type=int, default=200)
    parser.add_argument("--update-after", type=int, default=100)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--guided-episodes", type=int, default=200)
    parser.add_argument("--guided-exploration-prob", type=float, default=0.40)
    parser.add_argument("--warmup-noise-xy", type=float, default=0.18)
    parser.add_argument("--warmup-noise-charge", type=float, default=0.08)
    parser.add_argument("--warmup-noise-service", type=float, default=0.12)
    parser.add_argument("--warmup-noise-ratio", type=float, default=0.08)
    parser.add_argument("--guided-noise-xy", type=float, default=0.10)
    parser.add_argument("--guided-noise-charge", type=float, default=0.05)
    parser.add_argument("--guided-noise-service", type=float, default=0.08)
    parser.add_argument("--guided-noise-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="rl/outputs/training/logs")
    parser.add_argument("--plot-output-dir", type=str, default="rl/outputs/training/plots")
    parser.add_argument("--model-output-dir", type=str, default="rl/outputs/training/models")
    parser.add_argument("--auto-plot", dest="auto_plot", action="store_true")
    parser.add_argument("--no-auto-plot", dest="auto_plot", action="store_false")
    parser.set_defaults(auto_plot=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
