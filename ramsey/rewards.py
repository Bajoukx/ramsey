"""Reward functions for Ramsey environment."""

import abc
from typing import Callable, Dict

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
    graph_dict = env_utils.adj_vec_to_dict(env.adjacency_vec, env.n_vertices, color)
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

    def __init__(self, cumulative: bool = False):
        """Initializes the reward strategy."""
        self.cumulative = cumulative
        self.total_reward = 0.0

    def reset_total_reward(self):
        """Resets the total reward."""
        self.total_reward = 0.0

    @abc.abstractmethod
    def _compute_step_reward(self, obs, color):
        """Computes the reward given an observation and action."""
        pass

    def compute_reward(self, obs, color):
        """Computes the reward, updating total reward if cumulative."""
        reward, done, info = self._compute_step_reward(obs, color)
        if self.cumulative:
            self.total_reward += reward
        return reward, done, info


class SimpleRewardStrategy(RewardStrategy):
    """Simple reward strategy implementation."""
    def __init__(self,
                 max_clique_size,
                 reward_loss=-1.0,
                 terminal_reward_success=1.0,
                 cumulative: bool = False):
        """Initializes the simple reward strategy.
        
        This reward computes the reward for a single color. It penalizes each
        step with a negative reward until a monochromatic clique of size
        max_clique_size is found.

        Rewarding scheme:
            - creating monochromatic clique: terminal_reward_success and done
            - otherwise: -1 reward and continue
        """
        super().__init__(cumulative=cumulative)
        self.max_clique_size = max_clique_size
        self.reward_loss = reward_loss
        self.terminal_reward_success = terminal_reward_success
    
    def _compute_step_reward(self, obs, color):
        """Computes the simple reward.
        
        Takes the environment observation as the flattened adjacency vector.
        """
        graph_dict = env_utils.adj_vec_to_dict(obs, color)
        clique_list = clique_algorithms.bron_kerbosch(graph_dict)

        has_max_clique = False
        if clique_list:
            len_cliques = [len(clique) for clique in clique_list]
            if max(len_cliques) >= self.max_clique_size:
                has_max_clique = True

        if has_max_clique:
            done = True
            reward = self.terminal_reward_success
            return reward, done, {"violation_color": color}

        done = False
        return self.reward_loss, done, {}


################# WIP ####################


def hoffman_wip_reward(ramsey_env, action_color: int, regular_degree: int):
    """Computes Hoffman simple reward.
    
    Assuming that G is a d-regular graph, then the Hoffman bound states that for the independence number \alpha(G):
    \alpha(G) <= n * (-lambda_min) / (d - lambda_min)
    where lambda_min is the smallest eigenvalue of the adjacency matrix of G and \alpha is the independence number.

    The simple Hoffman reward function is defined as:
    f(G) = -avg_degree(G) + \beta * min(0, smallest_eigenvalue(G) - (n_vertives(G) * d / (n_vertices(G) - d)))
    
    In practice, we first create the adjacency matrix for the action_color, then we use torch to compute
    the smallest eigenvalue.
    """
    adj = torch.zeros((ramsey_env.n_vertices, ramsey_env.n_vertices),
                      dtype=torch.float)
    edge_indices = (ramsey_env.colored_edges == action_color).nonzero(
        as_tuple=True)[0]
    for idx in edge_indices:
        u, v = ramsey_env.all_edges[idx]
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    degrees = adj.sum(dim=1)

    avg_degree = degrees.mean().item()
    """if degrees.min().item() < regular_degree:
        # not a d-regular graph
        return -avg_degree, False, {"not_regular": True}"""

    # compute smallest eigenvalue
    eigenvalues = torch.linalg.eigvalsh(adj)
    smallest_eigenvalue = eigenvalues[0].real.item()

    beta = 1.0  # scaling factor for the eigenvalue term
    reward = -avg_degree + beta * min(
        0, smallest_eigenvalue - (ramsey_env.n_vertices * regular_degree /
                                  (ramsey_env.n_vertices - regular_degree)))
    #print('reward:', reward, 'avg_degree:', avg_degree, 'smallest_eigenvalue:', smallest_eigenvalue)

    # check terminal conditions
    found_max_clique = ramsey_env.has_max_clique(action_color,
                                                 ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        return reward, done, {"violation_color": action_color}

    return reward, False, {}


def hoffman_simple_reward(ramsey_env, action_color: int, regular_degree: int):
    """Computes the Hoffman simle reward.
    
    The Hoffman bound states that for a d-regular graph, the independence number \alpha(G) is bounded by:
    \alpha(G) <= n * (-lambda_min) / (d - lambda_min)
    where lambda_min is the smallest eigenvalue of the adjacency matrix of G.
    """
    adjacency_matrix = ramsey_env._edges_to_adjacency_tensor(action_color)
    eigenvalue_min = torch.linalg.eigvalsh(adjacency_matrix).min().item()
    # TODO: fix "The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 1).""
    mean_degree = adjacency_matrix.sum().item() / ramsey_env.n_vertices
    reward = ramsey_env.n_vertices * (-eigenvalue_min) / (mean_degree -
                                                          eigenvalue_min)

    # check terminal conditions
    found_max_clique = ramsey_env.has_max_clique(action_color,
                                                 ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        return reward, done, {"violation_color": action_color}
    return reward, False, {}


def max_eigenvalue_reward(ramsey_env, action_color: int):
    """Computes the maximum eigenvalue reward.

    The reward is defined as the maximum eigenvalue of the adjacency matrix of
    the graph formed by the edges of the given color.
    Based on: https://doi.org/10.1016/j.disc.2025.114694
    """
    adjacency_matrix = ramsey_env._edges_to_adjacency_tensor(action_color)
    max_eigenvalue = torch.linalg.eigvalsh(adjacency_matrix).max().item()
    reward = max_eigenvalue

    # check terminal conditions
    found_max_clique = ramsey_env.has_max_clique(action_color,
                                                 ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        return reward, done, {"violation_color": action_color}
    return reward, False, {}
