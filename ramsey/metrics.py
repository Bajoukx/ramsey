"""Metrics for undirected graphs for Ramsey numbers problem."""

def get_vertix_degree(adjacency_vec, n_vertices, vertex_idx):
    """Computes the degree of a vertex in an undirected graph.

    Args:
        adjacency_vec: 1D tensor representing the adjacency matrix of the graph.
        n_vertices: Number of vertices in the graph.
        vertex_idx: Index of the vertex whose degree is to be computed.
    Returns:
        Degree of the specified vertex.
    """
    degree = 0
    for j in range(n_vertices):
        if j == vertex_idx:
            continue
        edge_idx = (
            vertex_idx * n_vertices - (vertex_idx * (vertex_idx + 1)) // 2 +
            (j - vertex_idx - 1) if j > vertex_idx else
            j * n_vertices - (j * (j + 1)) // 2 + (vertex_idx - j - 1)
        )
        degree += adjacency_vec[edge_idx].item() != 0
    return degree


def get_average_degree(adjacency_vec, n_vertices, color=None):
    """Computes the average degree of vertices in an undirected graph.

    Args:
        adjacency_vec: 1D tensor representing the adjacency matrix of the graph.
        n_vertices: Number of vertices in the graph.
    Returns:
        Average degree of the vertices in the graph.
    """
    if not color is None:
        # Keep only entries equal to color; set others to 0
        adjacency_vec = [color if elm == color else 0 for elm in adjacency_vec]

    total_degree = 0
    for i in range(n_vertices):
        total_degree += get_vertix_degree(adjacency_vec, n_vertices, i)
    average_degree = total_degree / n_vertices
    return average_degree


def count_maximal_cliques(info_dict):
    """Counts the number of maximal cliques in dictionary of cliques.

    Args:
        clique_list: List of cliques, where each clique is represented as a list
        of vertex indices.
    Returns:
        Dictionary of number of maximal cliques by their sizes and color.
    """
    # Check if dictionary is empty
    if not "cliques_list" in info_dict:
        return {}
    else:
        counts = {}
        for color_key, clique_list in info_dict["cliques_list"].items():
            size_count = {}
            for clique in clique_list:
                size = len(clique)
                if size in size_count:
                    size_count[size] += 1
                else:
                    size_count[size] = 1
            counts[color_key] = size_count
        return counts
