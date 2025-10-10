"""Algorithms to find cliques in graph."""

import math

import torch

import operations


def bron_kerbosch(graph):
    """The Bron-Kerbosh Algorithm with pivot to find all maximal cliques.
    
    https://rosettacode.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    """

    cliques = []
    def _recursive(current_clique, candidates, processed_vertices, graph):
        if not candidates and not processed_vertices:
            if len(current_clique) > 2:
                cliques.append(list(current_clique))
            return

        union_set = candidates.union(processed_vertices)
        #print("union set:", union_set)
        pivot = max(union_set, key=lambda v: len(graph[v]))
        #print("pivot:", pivot)
        possibles = candidates.difference(graph[pivot])

        for vertex in possibles:
            new_clique = current_clique.union([vertex])
            #print("graph:", graph)
            #print("graph[vertex]:", graph[vertex])
            #print("candidates", candidates)
            new_candidates = candidates.intersection(graph[vertex])
            new_processed_vertices = processed_vertices.intersection(graph[vertex])
            _recursive(new_clique, new_candidates, new_processed_vertices, graph)
            candidates.remove(vertex)
            processed_vertices.add(vertex)
    _recursive(set(), set(graph.keys()), set(), graph)
    return cliques


def tensor_clique(adjacency_matrix, clique_size):
    """Counts the number of cliques of size `clique_size` in the graph represented by `adjacency_matrix` using the tensor clique representation.

    A tensor clique is first generated for the given adjacency matrix and then the number of cliques is counted based on the number of 1s in the tensor.
    """
    einsum_equation, einsum_matrices = operations.get_einsum_params(adjacency_matrix, clique_size)
    tensor_clique = torch.einsum(einsum_equation, *einsum_matrices)
    clique_count = tensor_clique.sum().item() // math.factorial(clique_size)
    return clique_count