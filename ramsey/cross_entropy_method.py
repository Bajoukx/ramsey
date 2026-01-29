"""Cross-Entropy Method (CEM) implementations for Ramsey graph search.

This module provides core CEM components for finding Ramsey graph colorings:
- Trajectory data structure for storing rollouts
- Policy networks for action selection
- Population collection and elite selection
- Training utilities for supervised learning on elite trajectories
- Circulant graph utilities for restricted search spaces
"""

from dataclasses import dataclass
from typing import List

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
        probs = self.get_action_probs(x)
        action = torch.multinomial(probs, num_samples=1).item()
        return action


def collect_trajectory(
    env,
    policy: PolicyNetwork,
    device: str = "cpu",
) -> Trajectory:
    """Collect a single trajectory by running policy in environment.

    Args:
        env: A Gymnasium-compatible environment with reset() and step().
        policy: The policy network for action selection.
        device: Device for tensor operations.

    Returns:
        A Trajectory containing observations, actions, and final score.
    """
    observations = []
    actions = []

    obs, _ = env.reset()
    done = False

    while not done:
        obs_tensor = obs.float().unsqueeze(0).to(device)
        observations.append(obs.clone())

        with torch.no_grad():
            action = policy.sample_action(obs_tensor)

        actions.append(action)
        obs, reward, _, done, info = env.step(action)

    return Trajectory(observations=observations,
                      actions=actions,
                      score=reward,
                      info=info)


def collect_population(
    env,
    policy: PolicyNetwork,
    population_size: int,
    device: str = "cpu",
) -> List[Trajectory]:
    """Collect a population of trajectories.

    Args:
        env: A Gymnasium-compatible environment.
        policy: The policy network for action selection.
        population_size: Number of trajectories to collect.
        device: Device for tensor operations.

    Returns:
        List of Trajectory objects.
    """
    trajectories = []
    for _ in range(population_size):
        traj = collect_trajectory(env, policy, device)
        trajectories.append(traj)
    return trajectories


def select_elite_by_fraction(
    trajectories: List[Trajectory],
    elite_fraction: float,
) -> List[Trajectory]:
    """Select top-performing trajectories by fraction.

    Args:
        trajectories: List of trajectories to select from.
        elite_fraction: Fraction of trajectories to keep (0.0 to 1.0).

    Returns:
        List of elite trajectories sorted by score (descending).
    """
    sorted_trajectories = sorted(
        trajectories,
        key=lambda t: t.score,
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
