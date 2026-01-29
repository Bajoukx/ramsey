"""Environment rendering utils."""

import itertools

import matplotlib.pyplot as plt
import networkx


def render_graph_from_adj_vec(adjacency_vec, n_vertices):
    """Renders a graph from its adjacency vector.
    
    Args:
        adjacency_vec: Flattened upper triangular adjacency vector where values
            represent edge colors (-1=uncolored, 0=red, 1=blue).
        n_vertices: Number of vertices in the graph.
    """
    all_edges = list(itertools.combinations(range(n_vertices), 2))

    graph = networkx.Graph()
    graph.add_nodes_from(range(n_vertices))

    for idx, (u, v) in enumerate(all_edges):
        c = adjacency_vec[idx]
        if hasattr(c, "item"):
            c = c.item()
        if c == -1:
            edge_color = "gray"
        elif c == 0:
            edge_color = "red"
        elif c == 1:
            edge_color = "blue"
        else:
            raise ValueError(f"Unknown color value '{c}' in adjacency vector.")
        graph.add_edge(u, v, color=edge_color)

    pos = networkx.circular_layout(graph)
    edge_colors = [graph[u][v]["color"] for u, v in graph.edges()]
    networkx.draw(graph,
                  pos,
                  with_labels=True,
                  edge_color=edge_colors,
                  node_color="lightgray",
                  node_size=500)
    plt.show()


def static_render(env):
    """Renders a static image of the graph."""
    n_vertices = env.n_vertices
    adjacency_vec = env.adjacency_vec

    graph = networkx.Graph()
    graph.add_nodes_from(range(n_vertices))
    colors = []
    for idx, (u, v) in enumerate(env.all_edges):
        c = adjacency_vec[idx].item()
        if c == -1:
            edge_color = "gray"
        elif c == 0:
            edge_color = "red"
        elif c == 1:
            edge_color = "blue"
        else:
            raise ValueError(f"Unknown color value '{c}' in adjacency vector.")
        graph.add_edge(u, v, color=edge_color)
        colors.append(edge_color)

    pos = networkx.spring_layout(graph)
    edge_colors = [graph[u][v]["color"] for u, v in graph.edges()]
    networkx.draw(graph,
                  pos,
                  with_labels=True,
                  edge_color=edge_colors,
                  node_color="lightgray",
                  node_size=500)
    plt.show()
