"""Gym environment wrapper for Ramsey problem."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Type, Union
import abc
import copy

import gymnasium
import numpy as np
import torch

from ramsey import action_types
from ramsey import ramsey_env
from ramsey import rendering
from ramsey import rewards


@dataclass
class Trajectory:
    """A single trajectory (construction) from the environment.

    Stores the sequence of observations and actions taken during an episode,
    along with the final score achieved.

    Args:
        observations: List of observations at each step.
        actions: List of actions taken at each step.
        rewards: List of rewards received at each step.
    """
    observations: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    info: dict = None

    def add_step(self, observation: torch.Tensor, action: int, reward: float,
                 info: dict):
        """Appends a single step's data to the trajectory."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.info = info


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

        self.observation_space = gymnasium.spaces.Box(low=-1,
                                                      high=self.n_colors - 1,
                                                      shape=(self.n_edges,),
                                                      dtype=np.float32)

        self.trajectory = Trajectory(observations=[], actions=[], rewards=[])
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]

    @property
    @abc.abstractmethod
    def action_space(self):
        """Decodes an action integer into its components."""
        raise NotImplementedError

    @abc.abstractmethod
    def truncate_episode(self) -> bool:
        """Truncates the episode if a maximum step count is reached.
        
        Default behavior is to not truncate.
        """
        return False

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        """Reset the environment.

        Resets the tracked episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        del options
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)

        if self.env.reward_strategy is not None:
            self.env.reward_strategy.total_reward = 0.0
            self.env.reward_strategy.reset_reward_info()

        observation, info = self.env.reset()
        # First observation has no associated action or reward
        self.trajectory = Trajectory(observations=[], actions=[], rewards=[])
        return observation, info

    def step(self, action: int):
        """Take a step in the environment.
        
        Keeps track of episode rewards. Flattens the adjacency matrix for
        compatibility and performance.
        """
        action = self._coerce_action_to_int(action)
        observation, reward, done, info = self.env.step(action)
        self.trajectory.add_step(observation, action, reward, info)
        truncated = self.truncate_episode()
        return observation, reward, done, truncated, info

    @staticmethod
    def _coerce_action_to_int(action) -> int:
        """Convert scalar-like actions to Python int.

        Handles native ints, NumPy scalar arrays, and scalar torch tensors.
        """
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(
                    "Action ndarray must be scalar-like (size == 1)")
            return int(action.reshape(-1)[0])

        if torch.is_tensor(action):
            if action.numel() != 1:
                raise ValueError(
                    "Action tensor must be scalar-like (numel == 1)")
            return int(action.item())

        return int(action)

    def render(self, mode: str = "None"):
        """Renders the environment."""
        if mode == "None":
            return
        elif mode == "static":
            rendering.static_render(self.env)
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

    def truncate_episode(self) -> bool:
        """Truncates the episode if a maximum step count is reached.
        
        Here, we set the maximum steps to be the number of edges *
        number of colors.
        """
        max_steps = self.n_edges * self.n_colors
        return self.env.steps >= max_steps


class RamseyGymEnvV1(BaseRamseyGymEnv):
    """Gym wrapper for RamseyEnv.
    
    The V1 version uses a size n_colors action space and a n_colors + n_edges
    observation space. The observation space space corresponds to the edge
    coloring and a one-hot encoding of the current vertex index to be colored.
    The action space corresponds to which color to assign to the current vertex.
    The edge index is inferred from the observation.
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
        """Overrides the default action space.
        
        The new observation space now represents the edge coloring and a one-hot
        encoding of the current vertex index.
        """
        action_strategy = action_types.TwoActionStrategy()
        super().__init__(n_vertices=n_vertices,
                         clique_sizes=clique_sizes,
                         init_method_name=init_method_name,
                         reward_strategy=reward_strategy,
                         action_strategy=action_strategy,
                         init_params=init_params,
                         render_mode=render_mode,
                         device=device)
        self.observation_space = gymnasium.spaces.Box(low=-1,
                                                      high=self.n_colors - 1,
                                                      shape=(self.n_colors + \
                                                      self.n_edges,),
                                                      dtype=np.float32)

    @property
    def action_space(self):
        action_dim = 2
        return gymnasium.spaces.Discrete(action_dim)

    def truncate_episode(self) -> bool:
        """V1 does not truncate episodes by default."""
        return False


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

    def truncate_episode(self) -> bool:
        """Truncates the episode if a maximum step count is reached.
        
        Here, we set the maximum steps to be the number of chord lengths *
        number of colors.
        """
        max_steps = self.env.n_chord_lengths * self.n_colors
        return self.env.steps >= max_steps


def make_env_factory(env_cls: Type[BaseRamseyGymEnv],
                     **env_kwargs) -> Callable[[], BaseRamseyGymEnv]:
    """Create a no-arg environment constructor for Gymnasium vector envs."""

    def _factory() -> BaseRamseyGymEnv:
        return env_cls(**copy.deepcopy(env_kwargs))

    return _factory


def make_sync_vector_env(env_cls: Type[BaseRamseyGymEnv], num_envs: int,
                         **env_kwargs) -> gymnasium.vector.SyncVectorEnv:
    """Construct a SyncVectorEnv for Ramsey Gym environment variants."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    env_fns = [make_env_factory(env_cls, **env_kwargs) for _ in range(num_envs)]
    return gymnasium.vector.SyncVectorEnv(env_fns)
