"""Tests for environment utils."""

import torch

from ramsey import env_utils


class TestUtils():
    """Test environment utils."""

    def test_flatten(self):
        """Test the adjacency matrix flattening."""
        adjacency_matrix = torch.Tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        flatened_matrix = env_utils.flaten_adjacency_matrix(adjacency_matrix)
        expected_flatened_matrix = torch.Tensor([1, 2, 1])
        assert torch.equal(flatened_matrix, expected_flatened_matrix)

    def test_unflatten(self):
        """Test the unflatening of the flattened adjacency matrix."""
        flatened_matrix = torch.tensor([1., 0., 1., 1., 0., 1.])
        unflatened_matrix = env_utils.unflaten_vec_to_adjacency_matrix(
            flatened_matrix)
        expected_unflatened_matrix = torch.tensor([[0., 1., 0., 1.],
                                                   [1., 0., 1., 0.],
                                                   [0., 1., 0., 1.],
                                                   [1., 0., 1., 0.]])
        assert torch.equal(unflatened_matrix, expected_unflatened_matrix)

    def test_decode_action(self):
        """Test action decoding."""

        class DummyEnv:
            n_colors = 3
            n_edges = 3
            n_vertices = 3

        env = DummyEnv()
        action = 7
        color, edge_idx = env_utils.decode_action(env, action)
        expected_edge_idx = 1
        expected_color = 2
        assert edge_idx == expected_edge_idx
        assert color == expected_color
