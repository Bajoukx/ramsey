"""Reward functions for Ramsey environment."""

import abc
from typing import Optional, Union

import torch

from ramsey import clique_algorithms
from ramsey import env_utils


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
            "max_clique_size": {},
            "is_counterexample": False
        }

    def reset_reward_info(self):
        """Resets the reward strategy state."""
        self.info = {
            "cliques_lists": {},
            "max_clique_size": {},
            "is_counterexample": False
        }

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
        super().__init__(cumulative=cumulative, reward_colors=reward_colors)
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
                info_max_clique_size = max(len_cliques)
                self.info["max_clique_size"][
                    f"color_{color}"] = info_max_clique_size
                if info_max_clique_size >= self.max_clique_size:
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
        super().__init__(cumulative=cumulative, reward_colors=reward_colors)
        self.max_clique_sizes = max_clique_sizes
        self.reward_loss = reward_loss
        self.reward_success = reward_success

    def _compute_step_reward(self, obs):
        """Computes the color sum reward."""
        self.reset_reward_info()
        total_reward = 0.0
        has_uncolored = torch.any(obs == -1).item()
        for color in self.reward_colors:
            graph_dict = env_utils.adj_vec_to_dict(obs, color)
            clique_list = clique_algorithms.bron_kerbosch(graph_dict)

            has_max_clique = False
            if clique_list:
                self.info["cliques_lists"][f"color_{color}"] = clique_list
                len_cliques = [len(clique) for clique in clique_list]
                max_clique_size = max(len_cliques)
                self.info["max_clique_size"][f"color_{color}"] = max_clique_size
                if max_clique_size >= self.max_clique_sizes[color]:
                    has_max_clique = True

            if not has_max_clique and not has_uncolored:
                total_reward += self.reward_success
            else:
                total_reward += self.reward_loss

        # Check if the graph is fully colored
        done = False
        if not has_uncolored:  # TODO: check for reserved -1 for uncolored
                               # instead of checking for uncolored.
            done = True
            found_counterexample = self._check_counterexample()
            if found_counterexample:
                self.info["is_counterexample"] = True

        return total_reward, done, self.info

    def _check_counterexample(self) -> bool:
        """Checks if a counterexample is found."""
        for color in self.reward_colors:
            clique_list = self.info["cliques_lists"].get(f"color_{color}", [])
            len_cliques = [len(clique) for clique in clique_list]
            # Check if there is a clique violating the max size
            if len_cliques and max(len_cliques) >= self.max_clique_sizes[color]:
                return False
        return True
