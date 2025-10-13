"""Utils for Ramsey."""

from typing import Optional, Union

import networkx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

def get_initial_random_coloring(n_edges: int, device: Optional[Union[str, torch.device]] = None):
    """Returns a tensor of shape [n_edges] filled with random 0 and 1 values."""
    max_edge_probability = 0.5  # TODO: Add probability as input
    edges = torch.rand(n_edges, device=device)
    edges = (edges < max_edge_probability).long()
    return edges

def get_initial_empty_coloring(n_edges: int, device: Optional[Union[str, torch.device]] = None):
    """Returns a tensor of shape [n_edges] filled with 0 values."""
    edges = torch.zeros(n_edges, dtype=torch.long, device=device)
    return edges
    
def static_render(n_vertices: int, all_edges: list, colored_edges: torch.Tensor):
    """Renders a static image of the graph."""
    G = networkx.Graph()
    G.add_nodes_from(range(n_vertices))
    n_red_vertices = (colored_edges == 0).sum().item()
    n_blue_vertices = (colored_edges == 1).sum().item()
    steps = n_red_vertices + n_blue_vertices

    colors = []
    for idx, (u, v) in enumerate(all_edges):
        c = colored_edges[idx].item()
        if c == -1:
            edge_color = "gray"
        elif c == 0:
            edge_color = "red"
        else:
            edge_color = "blue"
        G.add_edge(u, v, color=edge_color)
        colors.append(edge_color)


    pos = networkx.spring_layout(G)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]


    networkx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color="lightgray", node_size=500)
    plt.title(f"Ramsey R({n_red_vertices},{n_blue_vertices}) on K_{n_vertices}; step {steps}")
    plt.show()

def animated_render(n_vertices, n_red_vertices, n_blue_vertices, all_edges, colored_edges, steps, fig_ax=None):
    """Renders an animation of the graph. Only one window is created and updated."""
    # Check if fig_ax is None or missing keys
    if fig_ax is None or not all(k in fig_ax for k in ["_fig", "_ax", "_G", "_pos"]):
        fig, ax = plt.subplots(figsize=(6, 6))
        G = networkx.Graph()
        G.add_nodes_from(range(n_vertices))
        pos = networkx.spring_layout(G)
        ani = None

        networkx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=500, ax=ax)
        edges = networkx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax)
        labels = networkx.draw_networkx_labels(G, pos, ax=ax)
        ax.set_title(f"Ramsey R({n_red_vertices},{n_blue_vertices}) on K_{n_vertices}; step {steps}")
        plt.ion()
        plt.show(block=False)
        fig_ax = {"_fig": fig, "_ax": ax, "_G": G, "_pos": pos, "_ani": ani}
    else:
        fig = fig_ax["_fig"]
        ax = fig_ax["_ax"]
        G = fig_ax["_G"]
        pos = fig_ax["_pos"]

    G.clear_edges()
    colors = []
    for idx, (u, v) in enumerate(all_edges):
        c = colored_edges[idx].item()
        if c == -1:
            edge_color = "gray"
        elif c == 0:
            edge_color = "red"
        else:
            edge_color = "blue"
        G.add_edge(u, v, color=edge_color)
        colors.append(edge_color)

    ax.clear()
    networkx.draw(
        G, pos, ax=ax,
        with_labels=True, edge_color=colors,
        node_color="lightgray", node_size=500
    )
    ax.set_title(f"Ramsey R({n_red_vertices},{n_blue_vertices}) on K_{n_vertices}; step {steps}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig_ax