"""Algorithms to find cliques in graph."""

from ramsey import env_utils


def bron_kerbosch(graph_dict):
    """The Bron-Kerbosh Algorithm with pivot to find all maximal cliques.
    
    Returns the list of all maximal cliques.
    https://rosettacode.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    """
    cliques = []

    def _recursive(current_clique, candidates, processed_vertices, graph_dict):
        if not candidates and not processed_vertices:
            if len(current_clique) > 2:
                cliques.append(list(current_clique))
            return

        union_set = candidates.union(processed_vertices)
        pivot = max(union_set, key=lambda v: len(graph_dict[v]))
        possibles = candidates.difference(graph_dict[pivot])

        for vertex in possibles:
            new_clique = current_clique.union([vertex])
            new_candidates = candidates.intersection(graph_dict[vertex])
            new_processed_vertices = processed_vertices.intersection(
                graph_dict[vertex])
            _recursive(new_clique, new_candidates, new_processed_vertices,
                       graph_dict)
            candidates.remove(vertex)
            processed_vertices.add(vertex)

    _recursive(set(), set(graph_dict.keys()), set(), graph_dict)
    return cliques


def has_max_clique(env, color: int, size: int) -> bool:
    """Uses the Bron-Kerbosch algorithm to check if the maximal clique size is in the graph."""
    adjacency_dict = env._edjes_to_adj_dict(color=color)
    cliques = clique_algorithms.bron_kerbosch(adjacency_dict)
    if cliques:
        len_cliques = [len(clique) for clique in cliques]
        if max(len_cliques) >= size:
            return True
        if self.steps >= 135:  #TODO: remove
            print("max clique size:", max(len_cliques))
            print("graph:", self._edjes_to_adj_dict(color=color))
            print("found cliques:", cliques)
