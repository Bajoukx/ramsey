"""The ramsey environment."""

from typing import Optional, Tuple, List, Union
import itertools

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ramsey import env_utils
from ramsey import rewards


class RamseyEnv():
    """Gym-like environment for the Ramsey numbers problem."""

    def __init__(self,
                 n_vertices: int,
                 clique_sizes: List[int],
                 init_method_name: str = "empty",
                 reward_method_name: str = "simple",
                 init_params=None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize the Ramsey Environment."""

        if n_vertices < 1:
            raise ValueError("Invalid number of vertices.")

        self.n_vertices = n_vertices
        self.clique_sizes = clique_sizes
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.all_edges = list(itertools.combinations(range(n_vertices), 2))
        self.n_edges = len(self.all_edges)
        self.n_colors = len(self.clique_sizes)

        self.init_params = init_params or {}

        if init_method_name not in env_utils.get_all_init_methods():
            raise ValueError(f"Unknown init_method '{init_method_name}'")
        self.init_method_name = init_method_name
        self.init_function = env_utils.get_init_function(self.init_method_name)

        if reward_method_name not in rewards.get_all_reward_methods():
            raise ValueError(f"Unknown reward_method '{reward_method_name}'")
        self.reward_method_name = reward_method_name
        self.reward_function = rewards.get_reward_function(
            self.reward_method_name)

    def reset(self) -> torch.Tensor:
        """Resets environment."""
        self.adjacency_matrix = self.init_function(self, **self.init_params)
        self.done = False
        info = {}
        self.steps = 0
        return self.adjacency_matrix, info

    def step(self, action: int):
        """Apply action.
        
        An action is a color change in the graph i.e. the update of the
        adjacency matrix that represents the graph.

        Args:
            action: Integer representing the color change to make.
            reward_function: Callable function that attributes a value to
              action.
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call reset().")

        action_color, action_idx = env_utils.decode_action(self, action)
        self.adjacency_matrix[action_idx[0], action_idx[1]] = action_color
        self.adjacency_matrix[action_idx[1], action_idx[0]] = action_color

        reward, done, info = self.reward_function(
            self, action_color, self.clique_sizes[action_color],
            **self.init_params)
        return self.adjacency_matrix, reward, done, info
