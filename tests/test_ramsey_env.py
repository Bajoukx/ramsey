"""Test for Ramsey environment."""

import torch

from ramsey import ramsey_env


class TestRamseyEnv:
    """Test Ramsey environment."""

    _TEST_ENV = ramsey_env.RamseyEnv(n_vertices=5, clique_sizes=[3, 3])

    def test_number_of_edges(self):
        """Test if n_edges is correct."""
        n_edges = self._TEST_ENV.n_edges
        n_vertices = self._TEST_ENV.n_vertices
        expected_n_edges = n_vertices * ( n_vertices - 1) / 2
        assert n_edges == expected_n_edges

    def test_reset(self):
        """Test reset method."""
        matrix_shape = (self._TEST_ENV.n_edges, self._TEST_ENV.n_edges)
        init_matrix = torch.ones(matrix_shape)
        assert torch.equal(self._TEST_ENV.reset(init_matrix), init_matrix)
