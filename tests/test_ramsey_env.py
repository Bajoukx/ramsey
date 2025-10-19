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
        expected_n_edges = n_vertices * (n_vertices - 1) / 2
        assert n_edges == expected_n_edges

    def test_empty_initialization(self):
        """Test empty initialization, adjacency matrix filled with zeros."""
        env = ramsey_env.RamseyEnv(n_vertices=3,
                                   clique_sizes=[3, 3],
                                   init_method_name="empty")
        env_init_matrix, _ = env.reset()
        expected_zero_matrix = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert torch.equal(env_init_matrix, expected_zero_matrix)

    def test_step_action(self):
        """Test ajacency matrix updtade through step action."""
        action = 3
        env = ramsey_env.RamseyEnv(n_vertices=3,
                                   clique_sizes=[3, 3],
                                   init_method_name="empty")
        env.reset()
        adj_matrix, _, _, _ = env.step(action)
        expected_adjacency_matrix = torch.Tensor([[0, 1, 0], [1, 0, 0],
                                                  [0, 0, 0]])
        assert torch.equal(adj_matrix, expected_adjacency_matrix)
    
    def test_simple_reward(self):
        """Test Ramsey enviroment with simple reward."""
        env = ramsey_env.RamseyEnv(n_vertices=3,
                                   clique_sizes=[3, 3],
                                   init_method_name="empty",
                                   reward_method_name="simple")
        env.reset()
        actions = [3, 4]
        for action in actions:
            env.step(action)
        _, reward, done, _ = env.step(5)
        assert done is True 
        assert reward == 1.0
