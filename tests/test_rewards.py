"""Test for reward functions."""

import torch

from ramsey import rewards


class TestRewards:
    """Test for reward functions."""

    def test_simple_reward_no_cliques(self):
        """Test simple reward for no cliques found."""

        class DummyEnv:

            def __init__(self):
                self.adjacency_vec = torch.Tensor([1, 0, 0])
                self.n_vertices = 3

        env = DummyEnv()
        reward, done, info = rewards.simple_reward(env,
                                                   color=1,
                                                   max_clique_size=3,
                                                   reward_loss=-1.0,
                                                   terminal_reward_success=1.0)
        assert reward == -1.0
        assert done is False

    def test_simple_reward_clique(self):
        """Test simple reward when a clique is formed."""

        class DummyEnv:

            def __init__(self):
                self.adjacency_vec = torch.Tensor([1, 1, 1])
                self.n_vertices = 3

        env = DummyEnv()
        reward, done, info = rewards.simple_reward(env,
                                                   color=1,
                                                   max_clique_size=3)
        assert reward == 1.0
        assert done is True
