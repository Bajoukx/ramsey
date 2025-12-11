"""Gym environment wrapper for Ramsey problem."""

from typing import List, Optional, Union

import gymnasium
import torch
from typing import Optional, Union

from ramsey import env_utils
from ramsey import ramsey_env
from ramsey import rewards


class RamseyGymEnv(ramsey_env.RamseyEnv, gymnasium.Env):
    """Gym wrapper for RamseyEnv using PyG Data states."""

    metadata = {"render_modes": ["static", "animated", "None"]}

    def __init__(self,
                 n_vertices: int,
                 clique_sizes: List[int],
                 init_method_name: str = "empty",
                 reward_strategy: rewards.RewardStrategy = None,
                 init_params=None,
                 render_mode: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 ) -> None:
        super().__init__(n_vertices=n_vertices,
                         clique_sizes=clique_sizes,
                         init_method_name=init_method_name,
                         reward_strategy=reward_strategy,
                         init_params=init_params,
                         device=device)
        """Initializes the Ramsey Gym environment."""
        self.env = ramsey_env.RamseyEnv(n_vertices=n_vertices,
                                        clique_sizes=clique_sizes,
                                        init_method_name=init_method_name,
                                        init_params=init_params,
                                        reward_strategy=reward_strategy,
                                        device=device)
        
        self.action_space = gymnasium.spaces.Discrete(self.env.n_edges * self.env.n_colors)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=self.n_colors - 1,
            shape=(self.n_edges, ),
            dtype=int,
        )
        
        self.episode_rewards = []
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Resets the tracked episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, info = super().reset()
        return observation, info

    def step(self, action: int):
        """Take a step in the environment.
        
        Keeps track of episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, reward, done, info = super().step(action)
        self.episode_rewards.append(reward)
        truncated = False  # No time limits
        return observation, reward, truncated, done, info

    def render(self, mode: str = "None"):
        """Renders the environment."""
        if mode == "None":
            return
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' is not implemented yet.")