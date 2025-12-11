"""Test for reward functions."""

import torch

from ramsey import rewards


class TestRewards:
    """Test for reward functions."""

    _simple_reward_strategy = rewards.SimpleRewardStrategy(
            max_clique_size=3,
            reward_loss=-1.0,
            terminal_reward_success=1.0)

    def test_simple_reward_no_cliques(self):
        """Test simple reward for no cliques found."""
        adjacency_vec = torch.Tensor([1, 0, 0])
        reward, done, info = self._simple_reward_strategy.compute_reward(
            adjacency_vec, color=1)
        assert reward == -1.0
        assert done is False

    def test_simple_reward_clique(self):
        """Test simple reward when a clique is formed."""
        adjacency_vec = torch.Tensor([1, 1, 1])
        reward, done, info = self._simple_reward_strategy.compute_reward(
            adjacency_vec, color=1)
        assert reward == 1.0
        assert done is True

    def test_simple_reward_cumulative(self):
        """Test simple reward with cumulative=True."""
        cumulative_reward_strategy = rewards.SimpleRewardStrategy(
            max_clique_size=3,
            reward_loss=-1.0,
            terminal_reward_success=1.0,
            cumulative=True)

        adjacency_vec = torch.Tensor([1, 1, 1])
        reward, done, _ = cumulative_reward_strategy.compute_reward(
            adjacency_vec, color=1)
        assert reward == 0.0  # -1.0 + 1.0
        assert done is True
