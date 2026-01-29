"""Reward functions for Ramsey environment."""

import abc
from typing import Callable, Dict, Optional, Union

import torch

from ramsey import clique_algorithms
from ramsey import env_utils


def simple_reward(env,
                  color: int,
                  max_clique_size: int,
                  reward_loss: float = -1.0,
                  terminal_reward_success: float = 1.0):
    """Computes simple reward.

    This reward computes the reward for a single color. It penalizes each step
    with a negative reward until a monochromatic clique of size max_clique_size
    is found.

    Colors are represented as integers starting from 0.

    Rewarding scheme:
        - creating monochromatic clique: terminal_reward_success and done
        - otherwise: -1 reward and continue
    """
    graph_dict = env_utils.adj_vec_to_dict(env.adjacency_vec, color)
    clique_list = clique_algorithms.bron_kerbosch(graph_dict)

    has_max_clique = False
    if clique_list:
        len_cliques = [len(clique) for clique in clique_list]
        if max(len_cliques) >= max_clique_size:
            has_max_clique = True

    if has_max_clique:
        done = True
        reward = terminal_reward_success
        return reward, done, {"violation_color": color}

    done = False
    return reward_loss, done, {}


def get_all_reward_methods() -> Dict:
    """Get a dictionary with all pairs {reward_method: function}."""
    return {"simple": simple_reward}


def get_reward_function(method: str) -> Callable:
    """Gets the init function."""
    init_methods = get_all_reward_methods()
    return init_methods[method]


class RewardStrategy(abc.ABC):
    """Abstract base class for reward strategies."""

    def __init__(self,
                 cumulative: bool = False,
                 reward_colors: Optional[Union[list, int]] = None):
        """Initializes the reward strategy."""
        self.cumulative = cumulative
        self.total_reward = 0.0
        self.reward_colors = reward_colors

        self.info = {
            "cliques_lists": {},
            "max_clique_sizes": {}
        }

    def reset_total_reward(self):
        """Resets the total reward."""
        self.total_reward = 0.0

    @abc.abstractmethod
    def _compute_step_reward(self, obs):
        """Computes the reward given an observation and action."""
        pass

    def compute_reward(self, obs):
        """Computes the reward, updating total reward if cumulative."""
        reward, done, info = self._compute_step_reward(obs)
        if self.cumulative:
            self.total_reward += reward
            return self.total_reward, done, info
        return reward, done, info

    @abc.abstractmethod
    def _check_counterexample(self) -> bool:
        """Checks if the observation is a counterexample."""
        raise NotImplementedError



class SimpleRewardStrategy(RewardStrategy):
    """Simple reward strategy implementation."""

    def __init__(self,
                 max_clique_size,
                 reward_loss=-1.0,
                 terminal_reward_success=1.0,
                 cumulative: bool = False,
                 reward_colors: Optional[Union[list, int]] = None):
        """Initializes the simple reward strategy.
        
        This reward computes the reward for both colors. It penalizes each step
        with a reward loss until a coloring is found without any
        monochromatic clique of size max_clique_size.

        E.g. for a graph with 5 nodes and max_clique_size=3, if the 0-coloring
        has a triangle (3-clique) but the 1-coloring does not, the reward is
        reward_loss and the episode continues. If both colorings have no
        triangle, the reward is terminal_reward_success and the episode ends.

        Rewarding scheme:
            - creating monochromatic clique: reward_loss and continue
            - no monochromatic clique in all reward_colors:
              terminal_reward_success and done
        """
        super().__init__(cumulative=cumulative,
                         reward_colors=reward_colors)
        self.max_clique_size = max_clique_size
        self.reward_loss = reward_loss
        self.terminal_reward_success = terminal_reward_success

    def _compute_step_reward(self, obs):
        """Computes the simple reward.
        
        Takes the environment observation as the flattened adjacency vector.
        """
        has_max_clique = False
        done = False
        for color in self.reward_colors:
            graph_dict = env_utils.adj_vec_to_dict(obs, color)
            clique_list = clique_algorithms.bron_kerbosch(graph_dict)

            self.info["cliques_lists"][f"color_{color}"] = clique_list

            if clique_list:
                len_cliques = [len(clique) for clique in clique_list]
                if max(len_cliques) >= self.max_clique_size:
                    has_max_clique = True

        # Check if the graph is fully colored
        if not torch.any(obs == -1).item():
            done = True
            if not has_max_clique:
                reward = self.terminal_reward_success
            else:
                reward = self.reward_loss
            return reward, done, self.info

        reward = self.reward_loss
        return reward, done, self.info

    def _check_counterexample(self) -> bool:
        """Checks if the observation is a counterxample."""
        cliques_lists = self.info["cliques_lists"]
        has_max_clique = []
        if cliques_lists:
            for clique_list in cliques_lists.values():
                len_cliques = [len(clique) for clique in clique_list]
                has_max_clique.append(max(len_cliques) if len_cliques else 0)
        if all(size < self.max_clique_size for size in has_max_clique):
            return True
        return False


class ColorSumRewardStrategy(RewardStrategy):
    """Color sum reward strategy implementation."""

    def __init__(self,
                 max_clique_sizes,
                 reward_loss=-0.0,
                 reward_success=1.0,
                 cumulative: bool = False,
                 reward_colors: Optional[Union[list, int]] = None):
        """Initializes the Reward for summing the reward for each color.
        
        This reward attributes a reward_success value for any colors that does
        not contain a clique of the respective max_clique_size.

        E.g. for a 8 vertix graph with max_clique_sizes of [3, 4], in case it
        has a maximal clique of size 3 in color 0 but no maximal clique of size
        4 in color 1, then it receives a reward of reward_loss + reward_success.

        Assumes colors are represented as integers starting from 0, non-colored
        as -1.
        """
        super().__init__(cumulative=cumulative,
                         reward_colors=reward_colors)
        self.max_clique_sizes = max_clique_sizes
        self.reward_loss = reward_loss
        self.reward_success = reward_success

    def _compute_step_reward(self, obs):
        """Computes the color sum reward."""
        total_reward = 0.0
        for color in self.reward_colors:
            graph_dict = env_utils.adj_vec_to_dict(obs, color)
            clique_list = clique_algorithms.bron_kerbosch(graph_dict)

            has_max_clique = False
            if clique_list:
                self.info["cliques_lists"][f"color_{color}"] = clique_list
                len_cliques = [len(clique) for clique in clique_list]
                if max(len_cliques) >= self.max_clique_sizes[color]:
                    has_max_clique = True

            if not has_max_clique:
                total_reward += self.reward_success
            else:
                total_reward += self.reward_loss

        # Check if the graph is fully colored
        done = False
        if not torch.any(obs == -1).item():
            done = True
            if self._check_counterexample():
                self.info["is_counterexample"] = True

        return total_reward, done, self.info

    def _check_counterexample(self) -> bool:
        """Checks if a counterexample is found.
        
        Can only be checked if graph is fully colored.
        """
        for color in self.reward_colors:
            clique_list = self.info["cliques_lists"].get(f"color_{color}", [])
            len_cliques = [len(clique) for clique in clique_list]
            if len_cliques and max(len_cliques) >= self.max_clique_sizes[color]:
                return False
        return True
