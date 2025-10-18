"""The ramsey environment."""

from collections.abc import Callable
from typing import Optional, Tuple, List, Union
import itertools

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ramsey import env_utils


class RamseyEnv():
    """Gym-like environment for the Ramsey numbers problem."""

    def __init__(self,
                 n_vertices: int,
                 clique_sizes: List,
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

        self.reset()

    def reset(self, init_method: Callable) -> torch.Tensor:
        """Resets environment."""
        self.adjacency_matrix = init_method()
        self.done = False
        self.info = {}
        self.steps = 0
        return self.adjacency_matrix

    def step(self, action: Union[int, Tuple[int, int]],
             reward_function: Callable):
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
        
        self.adjacency_matrix = env_utils.apply_action(action)
        reward, done, info = reward_function(self.adjacency_matrix)
        return self.adjacency_matrix, reward, done, info

    