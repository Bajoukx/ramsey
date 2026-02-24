"""Cross-Entropy Method (CEM) implementations for Ramsey graph search.

This module provides core CEM components for finding Ramsey graph colorings:
- Trajectory data structure for storing rollouts
- Policy networks for action selection
- Population collection and elite selection
- Training utilities for supervised learning on elite trajectories
- Circulant graph utilities for restricted search spaces
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn


@dataclass
class Trajectory:
    """A single trajectory (construction) from the environment.

    Stores the sequence of observations and actions taken during an episode,
    along with the final score achieved.

    Args:
        observations: List of observations at each step.
        actions: List of actions taken at each step.
        score: Final score (reward) of the trajectory.
    """
    observations: List[torch.Tensor]
    actions: List[int]
    score: float
    info: dict = None


class PolicyNetwork(nn.Module):
    """Neural network policy for CEM.

    A feedforward network that maps observations to action logits.
    Uses dropout for regularization during training.

    Args:
        obs_dim: Size of observation space.
        action_dim: Number of possible actions.
        hidden_size: Size of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.2,
    ):
        """Initialize the policy network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits.

        Args:
            x: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Action logits of shape (batch_size, action_dim).
        """
        return self.net(x)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from observation.

        Args:
            x: Observation tensor.

        Returns:
            Action probabilities (softmax of logits).
        """
        logits = self.forward(x)
        # explicit softmax (equivalent)
        return torch.softmax(logits, dim=-1)

    def sample_action(self, x: torch.Tensor) -> int:
        """Sample action according to policy probability distribution.

        Args:
            x: Observation tensor (single observation).

        Returns:
            Sampled action index.
        """
        return int(self.sample_actions(x)[0].item())

    def sample_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one action per observation in a batch.

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Tensor of sampled action indices of shape (batch,).
        """
        probs = self.get_action_probs(x)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _obs_to_tensor(obs, device: str) -> torch.Tensor:
    """Convert observations to float tensor with batch dimension."""
    obs_tensor = torch.as_tensor(obs)
    if obs_tensor.dtype != torch.float32:
        obs_tensor = obs_tensor.float()
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor.to(device)


def collect_trajectory(env, policy: PolicyNetwork, device: str = "cpu"):
    """Collect a single trajectory by running policy in environment.

    Args:
        env: A Gymnasium-compatible environment with reset() and step().

    Returns:
        The environment after completing the trajectory.
    """
    obs, _ = env.reset()
    done = False

    while not done:
        obs_tensor = _obs_to_tensor(obs, device)
        with torch.no_grad():
            action = policy.sample_action(obs_tensor)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env


def collect_population(env,
                       policy: PolicyNetwork,
                       population_size: int,
                       device: str = "cpu",
                       num_envs: int = 1,
                       env_factory: Optional[Callable] = None):
    """Collect a population of environments.
    
    Args:
        env: A Gymnasium-compatible environment.
        policy: The policy network for action selection.
        population_size: Number of trajectories to collect.
        num_envs: Number of environments to run concurrently.
        env_factory: A no-argument environment constructor required when
            num_envs > 1.

    Returns:
        List of trajectories.
    """
    if population_size < 1:
        raise ValueError("population_size must be >= 1")

    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    if num_envs > 1:
        if env_factory is None:
            raise ValueError("env_factory must be provided when num_envs > 1")
        return collect_population_vectorized(
            env_factory=env_factory,
            policy=policy,
            population_size=population_size,
            num_envs=num_envs,
            device=device,
        )

    trajectories = []
    for _ in range(population_size):
        env_instance = env_factory() if env_factory is not None else env
        env_instance = collect_trajectory(env_instance, policy, device)
        trajectories.append(env_instance.trajectory)
    return trajectories


def collect_population_vectorized(env_factory: Callable,
                                  policy: PolicyNetwork,
                                  population_size: int,
                                  num_envs: int,
                                  device: str = "cpu"):
    """Collect trajectories in batches using batched policy inference.

    Each batch instantiates up to ``num_envs`` environments and advances them
    together by sampling all active actions in a single policy forward pass.
    """
    trajectories = []

    while len(trajectories) < population_size:
        batch_size = min(num_envs, population_size - len(trajectories))
        envs = [env_factory() for _ in range(batch_size)]
        observations = []
        active = [True] * batch_size

        for env_instance in envs:
            obs, _ = env_instance.reset()
            observations.append(obs)

        while any(active):
            active_indices = [
                i for i, is_active in enumerate(active) if is_active
            ]
            obs_batch = torch.cat(
                [
                    _obs_to_tensor(observations[idx], device)
                    for idx in active_indices
                ],
                dim=0,
            )
            with torch.no_grad():
                actions = policy.sample_actions(obs_batch).cpu().tolist()

            for action_idx, env_idx in enumerate(active_indices):
                env_instance = envs[env_idx]
                next_obs, _, terminated, truncated, _ = env_instance.step(
                    int(actions[action_idx]))
                observations[env_idx] = next_obs

                if terminated or truncated:
                    active[env_idx] = False
                    trajectories.append(env_instance.trajectory)

        for env_instance in envs:
            close_fn = getattr(env_instance, "close", None)
            if callable(close_fn):
                close_fn()

    return trajectories


def select_elite_by_fraction(trajectories, elite_fraction: float):
    """Select top-performing trajectories by fraction.

    Args:
        trajectories: List of trajectories to select from.
        elite_fraction: Fraction of trajectories to keep (0.0 to 1.0).
    Returns:
        List of elite trajectories sorted by score (descending).
    """
    sorted_trajectories = sorted(
        trajectories,
        key=lambda e: e.rewards[-1] if e.rewards else float("-inf"),
        reverse=True,
    )
    n_elite = max(1, int(len(trajectories) * elite_fraction))
    return sorted_trajectories[:n_elite]


def train_on_elite(
    policy: PolicyNetwork,
    elite_trajectories: List[Trajectory],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    supervised_steps: int,
    device: str = "cpu",
) -> float:
    """Train policy network on elite trajectories using supervised learning.

    Minimizes cross-entropy loss between predicted action logits and
    the actions taken in elite trajectories.

    Args:
        policy: The policy network to train.
        elite_trajectories: List of elite trajectories to learn from.
        optimizer: The optimizer for training.
        criterion: The loss function (typically CrossEntropyLoss).
        supervised_steps: Number of gradient steps to take.
        device: Device for tensor operations.

    Returns:
        Average loss over training steps.
    """
    all_obs = []
    all_actions = []
    for traj in elite_trajectories:
        for obs, action in zip(traj.observations, traj.actions):
            all_obs.append(obs)
            all_actions.append(action)

    if not all_obs:
        return 0.0

    obs_batch = torch.stack(all_obs).float().to(device)
    action_batch = torch.tensor(all_actions, dtype=torch.long, device=device)

    total_loss = 0.0
    for _ in range(supervised_steps):
        optimizer.zero_grad()
        logits = policy(obs_batch)
        loss = criterion(logits, action_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / supervised_steps
