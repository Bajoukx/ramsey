"""Defines action types for Ramsey environment."""

import abc


class BaseActionStrategy(abc.ABC):
    """Abstract base class for action strategies."""

    @abc.abstractmethod
    def decode_action(self, env, action: int):
        """Decode action into edge indices and color."""
        raise NotImplementedError


class DefaultActionStrategy(BaseActionStrategy):
    """Default action strategy using simple action decoding."""

    def decode_action(self, env, action: int):
        """Decodes an action into edge indices and color.
        
        An action is an integer in the set {0, 2, ..., n_edges * colors}. The
        first n_edges elements are assumed to be color 0, n_edges + 1 to 2 *
        n_edges color 1, and so on.
        """
        color = action // env.n_edges
        vec_color_idx = action % env.n_edges
        return color, vec_color_idx


class TwoActionStrategy(BaseActionStrategy):
    """Action strategy for n_colors action space."""
    def _infer_edge_idx(self, env):
        """Returns the edge index based on observation.
        
        This assumes that the observation corresponds to the edge coloring and a
        one-hot encoding of the current vertex index to be colored.
        """
        vertex_one_hot = env.adjacency_vec[-env.n_edges:]
        return vertex_one_hot.argmax().item()

    def decode_action(self, env, action: int):
        """Decodes an action into edge indices and color.
        
        An action is an integer in the set {0, n_colors}. This makes sure a
        color is always chosen.
        """
        edge_idx = self._infer_edge_idx(env)
        return action, edge_idx


class CirculantActionStrategy(BaseActionStrategy):
    """Action strategy for circulant graphs using chord length decoding."""

    def _chord_length_to_edge_indices(self, n_vertices: int, chord_length: int):
        """Converts a chord length to edge indices in the adjacency vector.
        
        In a circulant graph with n vertices, chord length k connects each
        vertex i to vertex (i + k) mod n. This function returns the indices of
        these edges in the flattened upper triangular adjacency vector.
        
        Args:
            n_vertices: Number of vertices in the graph. chord_length: The chord
            length (1 to floor(n/2)).
            
        Returns:
            Tensor of edge indices in the flattened upper triangular adjacency.
        """
        edge_indices = []

        for i in range(n_vertices):
            j = (i + chord_length) % n_vertices
            # Ensure i < j for upper triangular representation
            if i > j:
                i, j = j, i
            # Convert (i, j) to index in flattened upper triangular
            # Index = i * (2*n - i - 1) / 2 + (j - i - 1)
            idx = i * (2 * n_vertices - i - 1) // 2 + (j - i - 1)
            edge_indices.append(idx)

        return edge_indices

    def decode_action(self, env, action: int):
        """Decodes a chord length action into edge indices and color.
        
        A chord length action is an integer in the set {0, 1, ...,
        floor(env.n_edges / 2) * n_colors - 1}. The first n_chord_lengths
        elements are assumed to be color 0, the next n_chord_lengths color 1,
        and so on.
        """
        color = action // env.n_chord_lengths
        chord_length = action % env.n_chord_lengths + 1
        edge_idxs = self._chord_length_to_edge_indices(env.n_vertices,
                                                       chord_length)
        return color, edge_idxs
