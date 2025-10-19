"""Utils required for Ramsey Environment."""

import itertools
from typing import Callable, Dict

import torch


def flaten_adjacency_matrix(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Flatens a (n, n) adjacency matrix into a tensor size n*(n-1)/2.
    
    The first row is the first n-1 elements of the flattened adjacency matrix,
    the second row is the following n-2 elements, and so on.
    """
    n = adjacency_matrix.size(0)
    flatened = adjacency_matrix[torch.triu(torch.ones(n, n), diagonal=1) == 1]
    return flatened


def unflaten_to_adjacency_matrix(
        flatened_adjacency: torch.Tensor) -> torch.Tensor:
    """Unflatens a tensor size n*(n-1)/2 into a (n, n) adjacency matrix.

    The first n-1 elements in the unflatened are the first row of the adjacency
    matrix, the second row is the following n-2 elements, and so on. Finnally,
    we make the matrix symmetric.
    """
    length = flatened_adjacency.numel()
    n_vertices = int((1 + (1 + 8 * length)**0.5) / 2)

    adj = torch.zeros(n_vertices, n_vertices)
    idx = torch.triu_indices(n_vertices, n_vertices, offset=1)
    adj[idx[0], idx[1]] = flatened_adjacency
    adj[idx[1], idx[0]] = flatened_adjacency
    return adj


def adj_matrix_to_dict(adjacency_matrix: torch.Tensor, color: int):
    """Convert colored edges to adjacency dict for given color."""
    flatened_adjacency_matrix = flaten_adjacency_matrix(adjacency_matrix)
    colored_flatened_adjacency = (flatened_adjacency_matrix == color)
    colored_edges_idx = colored_flatened_adjacency.nonzero(as_tuple=True)[0]

    n_vertices = len(adjacency_matrix[0])
    all_possible_edges = list(itertools.combinations(range(n_vertices), 2))
    all_edges_index = torch.tensor(all_possible_edges).t().contiguous()

    adj_dict = {}
    for idx in colored_edges_idx:
        u, v = all_edges_index[:, idx].tolist()
        u, v = str(u), str(v)

        adj_dict.setdefault(u, set()).add(v)
        adj_dict.setdefault(v, set()).add(u)
    return adj_dict


def decode_action(env, action: int):
    """Decodes an action into edge indices and color.
    
    An action is an integer in the set {0, 2, ..., n_edges * colors}. The first
    n_edges elements are assumed to be color 0, n_edges + 1 to 2 * n_edges
    color 1, and so on.
    """
    color = action // env.n_edges
    vec_color_idx = action % env.n_edges
    #color = action // (env.n_colors * env.n_edges)
    #vec_color_idx = action % env.n_vertices

    matrix_idxs = torch.triu_indices(env.n_vertices, env.n_vertices, offset=1)
    matrix_color_idx = matrix_idxs[:, vec_color_idx]
    return color, matrix_color_idx


def init_empty(env) -> torch.Tensor:
    """Returns a tensor of shape [n_edges, n_edges] filled with 0 values."""
    n_vertices = env.n_vertices
    return torch.zeros(n_vertices, n_vertices)


def get_all_init_methods() -> Dict:
    """Get a dictionary with all pairs {init_method: function}."""
    return {"empty": init_empty}


def get_init_function(method: str) -> Callable:
    """Gets the init function."""
    init_methods = get_all_init_methods()
    return init_methods[method]
