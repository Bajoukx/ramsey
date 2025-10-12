import gymnasium
import torch
from torch import Tensor
from typing import Optional, Tuple, Union, Dict, List

import ramsey_game
import utils

class RamseyGymEnv(gymnasium.Env):
    """Gym wrapper for RamseyEnv using PyG Data states."""

    metadata = {"render.modes": ["static", "animated"]}

    def __init__(
        self,
        n_vertices: int,
        n_red_edges: int,
        n_blue_edges: int,
        render_mode: Optional[str] = None,
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
        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

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
        truncated = False  # No time limits
        self.episode_rewards.append(reward)
        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == "static":
            utils.static_render(self.env.n_vertices, self.env.all_edges, self.env.colored_edges)
        elif self.render_mode == "animated":
            if not hasattr(self, "_fig_ax"):
                self._fig_ax = None
            self._fig_ax = utils.animated_render(self.env.n_vertices,
                                                 self.env.n_red_vertices,
                                                 self.env.n_blue_vertices,
                                                 self.env.all_edges,
                                                 self.env.colored_edges,
                                                 self.env.steps,
                                                 self._fig_ax)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not implemented.")
