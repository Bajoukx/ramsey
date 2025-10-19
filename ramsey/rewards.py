"""Reward functions for Ramsey environment."""

from typing import Callable, Dict

from ramsey import clique_algorithms
from ramsey import env_utils


def simple_reward(env,
                  color: int,
                  max_clique_size: int,
                  reward_loss: float = -1.0,
                  terminal_reward_success: float = 1.0):
    """Computes simple reward.

    Rewarding scheme:
        - creating monochromatic clique: terminal_reward_success and done
        - otherwise: -1 reward and continue
    """
    graph_dict = env_utils.adj_matrix_to_dict(env.adjacency_matrix, color)
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
