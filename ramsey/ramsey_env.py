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
                 reward_strategy: rewards.RewardStrategy = None,
                 init_params=None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize the Ramsey Environment."""
        if n_vertices < 1:
            raise ValueError("Invalid number of vertices.")

        self.n_vertices = n_vertices
        self.clique_sizes = clique_sizes
        self.device = device if device is not None else torch.device("cpu")

        self.all_edges = list(itertools.combinations(range(n_vertices), 2))
        self.n_edges = len(self.all_edges)
        self.n_colors = len(self.clique_sizes)

        self.init_params = init_params or {}

        if init_method_name not in env_utils.get_all_init_methods():
            raise ValueError(f"Unknown init_method '{init_method_name}'")
        self.init_method_name = init_method_name
        self.init_function = env_utils.get_init_function(self.init_method_name)

        self.reward_strategy = reward_strategy

    def reset(self) -> torch.Tensor:
        """Resets environment."""
        self.adjacency_vec = self.init_function(self, **self.init_params)
        self.done = False
        info = {}
        self.steps = 0
        return self.adjacency_vec.to(self.device), info

    def step(self, action: int):
        """Apply action.
        
        An action is a color change in the graph i.e. the update of the
        adjacency vector that represents the graph.

        Args:
            action: Integer representing the color change to make.
            reward_function: Callable function that attributes a value to
              action.
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call reset().")

        action_color, action_idx = env_utils.decode_action(self, action)
        self.adjacency_vec[action_idx] = action_color

        reward, done, info = self.reward_strategy.compute_reward(
            self.adjacency_vec, action_color)
        return self.adjacency_vec.to(self.device), reward, done, info
