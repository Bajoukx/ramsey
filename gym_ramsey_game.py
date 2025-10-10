import gymnasium
import torch
from torch import Tensor
from typing import Optional, Tuple, Union, Dict, List

import ramsey_game

class RamseyGymEnv(gymnasium.Env):
    """Gym wrapper for RamseyEnv using PyG Data states."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_vertices: int,
        n_red_edges: int,
        n_blue_edges: int,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()
        self.env = ramsey_game.RamseyEnv(
            n_vertices=n_vertices,
            n_red_edges=n_red_edges,
            n_blue_edges=n_blue_edges,
            device=device
        )
        self.n_edges = self.env.n_edges

        self.action_space = gymnasium.spaces.Discrete(self.n_edges * 2)  # edge x color
        self.observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_edges,),
            dtype=int,
        )
        self.episode_rewards = []

    def reset(self):
        info = {}
        obs = self.env.reset()
        self.episode_rewards = []
        return obs, info

    def step(self, action):#: Union[int, Tuple[int, int]]):
        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        obs = obs.reshape(-1)
        #obs = self._convert_obs(state)
        truncated = False  # No time limits
        self.episode_rewards.append(reward)
        print(self.episode_rewards)
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            self.env.render()
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented.")

    #def _convert_obs(self, state: torch.Tensor) -> np.ndarray:
    #    """Convert torch state (with -1,0,1) to numpy (0,1,2)."""
    #    arr = state.detach().cpu().numpy().astype(np.int32)
    #    return arr
