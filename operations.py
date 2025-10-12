"""Operations functions for Ramsey Graphs."""

from typing import List

import torch
from torch import Tensor


def get_einsum_params(adjacency_matrix: Tensor, clique_size: int) -> List[Tensor]:
    """Generates equation and the list of adjacency matrices to be used with torch.einsum."""
    if clique_size < 2:
        raise ValueError("Clique size must be at least 2.")

    indices = [chr(ord('i') + i) for i in range(clique_size)]
    terms = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            terms.append(f"{indices[i]}{indices[j]}")
    equation = ', '.join(terms) + ' -> ' + ''.join(indices)

    n_terms = clique_size * (clique_size - 1) // 2
    matrices = [adjacency_matrix] * n_terms
    return equation, matrices

def get_tensor_clique(adjacency_matrix: Tensor, clique_size: int) -> Tensor:
    """Generates the tensor clique representation of the graph for cliques of size `clique_size`."""
    einsum_equation, einsum_matrices = get_einsum_params(adjacency_matrix, clique_size)
    tensor_clique = torch.einsum(einsum_equation, *einsum_matrices)
    return tensor_clique
