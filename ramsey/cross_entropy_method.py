"""Cross-Entropy Method (CEM) implementations for Ramsey graph search.

This module provides core CEM components for finding Ramsey graph colorings:
- Trajectory data structure for storing rollouts
- Policy networks for action selection
- Population collection and elite selection
- Training utilities for supervised learning on elite trajectories
- Circulant graph utilities for restricted search spaces
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from ramsey import clique_algorithms
from ramsey import env_utils


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

    return Trajectory(
        observations=observations,
        actions=actions,
        score=reward,
        info=info
    )


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


def select_elite_by_percentile(
    trajectories: List[Trajectory],
    elite_percentile: float,
) -> List[Trajectory]:
    """Select top-performing trajectories by percentile threshold.

    Args:
        trajectories: List of trajectories to select from.
        elite_percentile: Percentile threshold (0.0 to 100.0).

    Returns:
        List of elite trajectories sorted by score (descending).
    """
    sorted_trajectories = sorted(
        trajectories,
        key=lambda t: t.score,
        reverse=True,
    )
    scores = [t.score for t in trajectories]
    threshold = torch.tensor(scores).quantile(elite_percentile / 100.0).item()
    elite = [t for t in sorted_trajectories if t.score >= threshold]
    return elite if elite else sorted_trajectories[:1]


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


def chord_lengths_to_adjacency_vec(
    n_vertices: int,
    chord_colors: torch.Tensor,
) -> torch.Tensor:
    """Convert chord length colors to flattened upper triangular adjacency.

    Circulant graphs are defined by chord lengths. Given numbered vertices,
    vertex i connects to vertices (i + k) mod n for each chord length k.

    Args:
        n_vertices: Number of vertices in the graph.
        chord_colors: Tensor of size (n_chord_lengths,) with color for each
            chord length. Chord length k is at index k-1.

    Returns:
        Flattened upper triangular adjacency vector.
    """
    n_edges = n_vertices * (n_vertices - 1) // 2
    adjacency_vec = torch.zeros(n_edges, dtype=torch.long)

    for chord_idx, color in enumerate(chord_colors):
        chord_length = chord_idx + 1
        edge_indices = env_utils.chord_length_to_edge_indices(
            n_vertices, chord_length)
        for idx in edge_indices:
            adjacency_vec[idx] = color.long()

    return adjacency_vec


def evaluate_graph(
    adjacency_vec: torch.Tensor,
    clique_sizes: List[int],
) -> Tuple[bool, int, dict]:
    """Evaluate a circulant graph for Ramsey violations.

    A Ramsey violation occurs if there exists a monochromatic clique of size
    greater than or equal to the specified maximum clique size for that color.

    Args:
        adjacency_vec: Flattened upper triangular adjacency vector.
        clique_sizes: List of maximum clique sizes for each color.

    Returns:
        Tuple of (is_valid, total_violations, violation_details).
        is_valid is True if no color has a monochromatic clique of the
        corresponding size.
    """
    total_violations = 0
    violation_details = {}

    for color, max_size in enumerate(clique_sizes):
        graph_dict = env_utils.adj_vec_to_dict(adjacency_vec, color)
        cliques = clique_algorithms.bron_kerbosch(graph_dict)

        max_clique_found = 0
        if cliques:
            max_clique_found = max(len(c) for c in cliques)

        if max_clique_found >= max_size:
            violations = max_clique_found - max_size + 1
            total_violations += violations
            violation_details[color] = {
                "max_clique": max_clique_found,
                "violations": violations
            }

    is_valid = total_violations == 0
    return is_valid, total_violations, violation_details


def score_circulant(
    chord_colors: torch.Tensor,
    n_vertices: int,
    clique_sizes: List[int],
) -> float:
    """Score a circulant graph configuration.

    Higher scores are better. Returns negative violations count,
    so the best score is 0 (no violations).

    Args:
        chord_colors: Tensor of colors for each chord length.
        n_vertices: Number of vertices.
        clique_sizes: Maximum clique sizes for each color.

    Returns:
        Score (negative of total violations).
    """
    adjacency_vec = chord_lengths_to_adjacency_vec(n_vertices, chord_colors)
    _, violations, _ = evaluate_graph(adjacency_vec, clique_sizes)
    return -violations


class CirculantCEM:
    """Cross-Entropy Method for Ramsey graph search.

    Args:
        n_vertices: Number of vertices in the graph.
        clique_sizes: Maximum clique sizes for each color.
        population_size: Number of samples per iteration.
        elite_fraction: Fraction of samples to use as elites.
        initial_prob: Initial probability for color 1 (vs color 0).
        learning_rate: How fast to update probabilities toward elite mean.
        device: Torch device.
    """

    def __init__(
        self,
        n_vertices: int,
        clique_sizes: List[int],
        population_size: int = 64,
        elite_fraction: float = 0.2,
        initial_prob: float = 0.5,
        learning_rate: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize CEM for circulant graphs."""
        self.n_vertices = n_vertices
        self.clique_sizes = clique_sizes
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_fraction))
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # Number of chord lengths is floor(n/2)
        self.n_chord_lengths = n_vertices // 2

        # prob[i] = probability that chord i+1 has color 1
        self.probs = torch.full(
            (self.n_chord_lengths, ),
            initial_prob,
            dtype=torch.float,
            device=self.device,
        )

        self.best_score = float("-inf")
        self.best_solution = None

    def sample_population(self) -> torch.Tensor:
        """Sample a population of chord colorings.

        Returns:
            Tensor of shape (population_size, n_chord_lengths) with colors.
        """
        samples = torch.bernoulli(
            self.probs.unsqueeze(0).expand(self.population_size, -1)).long()
        return samples

    def evaluate_population(
        self,
        population: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[bool]]:
        """Evaluate all samples in the population.

        Args:
            population: Tensor of shape (population_size, n_chord_lengths).

        Returns:
            Tuple of (scores tensor, list of validity flags).
        """
        scores = []
        valid_flags = []

        for i in range(self.population_size):
            chord_colors = population[i]
            score = score_circulant(chord_colors, self.n_vertices,
                                    self.clique_sizes)
            scores.append(score)
            valid_flags.append(score == 0)

        return torch.tensor(scores), valid_flags

    def update_probabilities(
        self,
        population: torch.Tensor,
        scores: torch.Tensor,
    ):
        """Update probabilities based on elite samples.

        Args:
            population: Tensor of shape (population_size, n_chord_lengths).
            scores: Tensor of scores for each sample.
        """
        # Get elite indices (highest scores)
        elite_indices = torch.argsort(scores, descending=True)[:self.n_elite]
        elite_samples = population[elite_indices]

        # Compute mean color for each chord among elites
        elite_mean = elite_samples.float().mean(dim=0)

        # Update probabilities with learning rate
        self.probs = ((1 - self.learning_rate) * self.probs +
                      self.learning_rate * elite_mean)

        # Clamp probabilities to avoid extremes
        self.probs = torch.clamp(self.probs, 0.01, 0.99)

        # Track best solution
        best_idx = elite_indices[0]
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx].item()
            self.best_solution = population[best_idx].clone()

    def run(self, num_iterations: int) -> Tuple[torch.Tensor, float, bool]:
        """Run CEM optimization.

        Args:
            num_iterations: Number of iterations to run.

        Returns:
            Tuple of (best_solution, best_score, found_valid).
        """
        found_valid = False

        for _ in range(num_iterations):
            # Sample population
            population = self.sample_population()

            # Evaluate
            scores, valid_flags = self.evaluate_population(population)

            # Update probabilities
            self.update_probabilities(population, scores)

            # Check for valid solutions
            if any(valid_flags):
                found_valid = True

            # Early stopping if we found valid solutions
            if self.best_score == 0:
                break

        return self.best_solution, self.best_score, found_valid

    def get_solution_chords(self) -> Tuple[List[int], List[int]]:
        """Get chord assignments for the best solution found.

        Returns:
            Tuple of (red_chords, blue_chords) where each is a list of
            chord lengths assigned to that color.
        """
        if self.best_solution is None:
            return [], []

        red_chords = [
            i + 1 for i, c in enumerate(self.best_solution) if c == 0
        ]
        blue_chords = [
            i + 1 for i, c in enumerate(self.best_solution) if c == 1
        ]
        return red_chords, blue_chords
