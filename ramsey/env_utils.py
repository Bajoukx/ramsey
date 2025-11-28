"""Utils required for Ramsey Environment."""

import itertools
from typing import Callable, Dict

import torch


def flaten_adjacency_matrix(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Flatens a (n, n) adjacency matrix into a tensor size n*(n-1)/2.
    
    The first row is the first n-1 elements of the flattened adjacency matrix,
    the second row is the following n-2 elements, and so on.
    Args:
        adjacency_matrix: A (n, n) adjacency matrix.
    Returns:
        A numpy array of size n*(n-1)/2 with dtype int64.
    """
    n = adjacency_matrix.size(0)
    flatened = adjacency_matrix[torch.triu(torch.ones(n, n), diagonal=1) == 1]
    return flatened.long()

def unflaten_vec_to_adjacency_matrix(
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


def adj_vec_to_dict(adjacency_vec: torch.Tensor, color: int) -> Dict[str, set]:
    """Convert colored edges to adjacency dict for given color. 
    
    Args:
        adjacency_vec: Flattened upper triangular adjacency vector.
        n_vertices: Number of vertices in the graph.
        color: Color value to filter edges by.
    
    Returns:
        Dictionary mapping vertex strings to sets of adjacent vertices.
    """
    n_vertices = int((1 + (1 + 8 * len(adjacency_vec))**0.5) / 2)
    colored_edges_idx = (adjacency_vec == color).nonzero(as_tuple=True)[0]
    
    all_edges = list(itertools.combinations(range(n_vertices), 2))
    
    adj_dict = {}
    for idx in colored_edges_idx:
        u, v = all_edges[idx]
        u_str, v_str = str(u), str(v)
        
        adj_dict.setdefault(u_str, set()).add(v_str)
        adj_dict.setdefault(v_str, set()).add(u_str)
    
    return adj_dict



def decode_action(env, action):
    """Decodes an action into edge indices and color.
    
    An action is an integer in the set {0, 2, ..., n_edges * colors}. The first
    n_edges elements are assumed to be color 0, n_edges + 1 to 2 * n_edges
    color 1, and so on.
    """
    color = action // env.n_edges
    vec_color_idx = action % env.n_edges
    return color, vec_color_idx


def init_empty(env) -> torch.Tensor:
    """Returns a vector of shape [n_edges] filled with 0's."""
    n_edges = env.n_edges
    return torch.zeros(n_edges)


def get_all_init_methods() -> Dict:
    """Get a dictionary with all pairs {init_method: function}."""
    return {"empty": init_empty}


def get_init_function(method: str) -> Callable:
    """Gets the init function."""
    init_methods = get_all_init_methods()
    return init_methods[method]
