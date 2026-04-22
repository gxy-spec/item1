from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean_layer(hidden)
        log_std = torch.clamp(self.log_std_layer(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        correction = torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q = mlp(obs_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=1))


@dataclass
class SACUpdateStats:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        self.actor = GaussianPolicy(obs_dim, action_dim, hidden_dim, action_low, action_high).to(device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(obs_tensor)
            else:
                action, _, _ = self.actor.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, torch.Tensor]) -> SACUpdateStats:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            next_q1 = self.target_critic1(next_obs, next_actions)
            next_q2 = self.target_critic2(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        sampled_actions, log_prob, _ = self.actor.sample(obs)
        q1_pi = self.critic1(obs, sampled_actions)
        q2_pi = self.critic2(obs, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return SACUpdateStats(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state": self.actor.state_dict(),
                "critic1_state": self.critic1.state_dict(),
                "critic2_state": self.critic2.state_dict(),
                "target_critic1_state": self.target_critic1.state_dict(),
                "target_critic2_state": self.target_critic2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic1.load_state_dict(checkpoint["critic1_state"])
        self.critic2.load_state_dict(checkpoint["critic2_state"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
