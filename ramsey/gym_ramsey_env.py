"""Gym environment wrapper for Ramsey problem."""

from typing import List, Optional, Union

import gymnasium
import torch
from torch import Tensor
from typing import Optional, Union

from ramsey import env_utils
from ramsey import ramsey_env


class RamseyGymEnv(gymnasium.Env):
    """Gym wrapper for RamseyEnv using PyG Data states."""

    metadata = {"render.modes": ["static", "animated", "None"]}

    def __init__(self,
                 n_vertices: int,
                 clique_sizes: List[int],
                 init_method_name: str = "empty",
                 reward_method_name: str = "simple",
                 init_params=None,
                 render_mode: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        """Initializes the Ramsey Gym environment."""
        self.env = ramsey_env.RamseyEnv(n_vertices=n_vertices,
                                        clique_sizes=clique_sizes,
                                        init_method_name=init_method_name,
                                        init_params=init_params,
                                        reward_method_name=reward_method_name,
                                        device=device)
        
        self.action_space = gymnasium.spaces.Discrete(self.env.n_edges)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=self.env.n_colors - 1,
            shape=(self.env.n_edges,),
            dtype=int,
        )
        # TODO: Support multiple players in the gym wrapper
        
        self.episode_rewards = []
        assert render_mode is None or render_mode in self.metadata[
            "render.modes"]

    def reset(self):
        """Reset the environment.

        Resets the tracked episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, info = self.env.reset()
        flat_observation = env_utils.flaten_adjacency_matrix(observation)
        self.episode_rewards = []
        return flat_observation, info

    def step(self, action: int):
        """Take a step in the environment.
        
        Keeps track of episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, reward, done, info = self.env.step(action)
        self.episode_rewards.append(reward)
        truncated = False  # No time limits
        flat_observation = env_utils.flaten_adjacency_matrix(observation)
        return flat_observation, reward, truncated, done, info

    def render(self, mode: str = "None"):
        """Renders the environment."""
        if mode == "None":
            return
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' is not implemented yet.")
        
