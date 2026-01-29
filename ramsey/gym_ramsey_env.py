"""Gym environment wrapper for Ramsey problem."""

from typing import List, Optional, Union
import abc

import gymnasium
import torch

from ramsey import action_types
from ramsey import ramsey_env
from ramsey import rewards


class BaseRamseyGymEnv(gymnasium.Env, abc.ABC):
    """Abstract base class for Ramsey Gym environments."""

    metadata = {"render_modes": ["static", "animated", "None"]}

    def __init__(
        self,
        n_vertices: int,
        clique_sizes: List[int],
        init_method_name: str = "empty",
        reward_strategy: rewards.RewardStrategy = None,
        action_strategy: action_types.BaseActionStrategy = None,
        init_params=None,
        render_mode: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.env = ramsey_env.RamseyEnv(n_vertices=n_vertices,
                                        clique_sizes=clique_sizes,
                                        init_method_name=init_method_name,
                                        init_params=init_params,
                                        reward_strategy=reward_strategy,
                                        action_strategy=action_strategy,
                                        device=device)
        self.n_edges = self.env.n_edges
        self.n_colors = self.env.n_colors

        self.observation_space = gymnasium.spaces.Box(low=0,
                                                      high=self.n_colors - 1,
                                                      shape=(self.n_edges,),
                                                      dtype=int)

        self.episode_rewards = []
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]

    @property
    @abc.abstractmethod
    def action_space(self):
        """Decodes an action integer into its components."""
        raise NotImplementedError

    def reset(self):
        """Reset the environment.

        Resets the tracked episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, info = self.env.reset()
        self.episode_rewards = []
        return observation, info

    def step(self, action: int):
        """Take a step in the environment.
        
        Keeps track of episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        observation, reward, done, info = self.env.step(action)
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


class RamseyGymEnvV0(BaseRamseyGymEnv):
    """Gym wrapper for RamseyEnv.
    
    The V0 version uses the default action strategy where each action
    corresponds to coloring an edge with a specific color.
    """

    def __init__(
        self,
        n_vertices: int,
        clique_sizes: List[int],
        init_method_name: str = "empty",
        reward_strategy: rewards.RewardStrategy = None,
        init_params=None,
        render_mode: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        action_strategy = action_types.DefaultActionStrategy()
        super().__init__(n_vertices=n_vertices,
                         clique_sizes=clique_sizes,
                         init_method_name=init_method_name,
                         reward_strategy=reward_strategy,
                         action_strategy=action_strategy,
                         init_params=init_params,
                         render_mode=render_mode,
                         device=device)

    @property
    def action_space(self):
        action_dim = self.n_edges * self.n_colors
        return gymnasium.spaces.Discrete(action_dim)


class RamseyGymEnvV1(BaseRamseyGymEnv):
    """Gym wrapper for RamseyEnv.
    
    The V1 version uses a 2 * n_colors action space where each action
    corresponds to coloring or doing nothing when coloring a graph by order.
    """

    metadata = {"render_modes": ["static", "animated", "None"]}

    def __init__(
        self,
        n_vertices: int,
        clique_sizes: List[int],
        init_method_name: str = "empty",
        reward_strategy: rewards.RewardStrategy = None,
        init_params=None,
        render_mode: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        action_strategy = action_types.TwoActionStrategy()
        super().__init__(n_vertices=n_vertices,
                         clique_sizes=clique_sizes,
                         init_method_name=init_method_name,
                         reward_strategy=reward_strategy,
                         action_strategy=action_strategy,
                         init_params=init_params,
                         render_mode=render_mode,
                         device=device)

    @property
    def action_space(self):
        action_dim = 2 * self.n_colors
        return gymnasium.spaces.Discrete(action_dim)


class RamseyGymEnvV2(BaseRamseyGymEnv):
    """Gym wrapper for RamseyEnv with circulant action strategy.
    
    Circulant graphs are defined by chord lengths. Given numered vertices, each
    vertix has an edge to the vertices at the specified chord lengths.

    Main difference is the action spaces. The action space is the number of
    possible chord lengths times the number of colors. A graph with n vertices
    can have floor(n/2) possible chord lengths.
    """

    def __init__(
        self,
        n_vertices: int,
        clique_sizes: List[int],
        init_method_name: str = "empty",
        reward_strategy: rewards.RewardStrategy = None,
        init_params=None,
        render_mode: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        action_strategy = action_types.CirculantActionStrategy()
        super().__init__(n_vertices=n_vertices,
                         clique_sizes=clique_sizes,
                         init_method_name=init_method_name,
                         reward_strategy=reward_strategy,
                         action_strategy=action_strategy,
                         init_params=init_params,
                         render_mode=render_mode,
                         device=device)
        self.env.n_chord_lengths = self.env.n_vertices // 2

    @property
    def action_space(self):
        action_dim = self.env.n_chord_lengths * self.n_colors
        return gymnasium.spaces.Discrete(action_dim)
