"""Tests for reward strategies in the Ramsey environment."""

import pytest
import torch

from ramsey.rewards import SimpleRewardStrategy, ColorSumRewardStrategy


class TestSimpleRewardStrategy:
    """Test suite for SimpleRewardStrategy."""

    def test_simple_reward_fully_colored_no_violation(self):
        """Test SimpleRewardStrategy when fully colored with no violations.

        Expected: terminal_reward_success (1.0), done=True
        """
        # Setup: 3-vertex graph, all colored, no triangle in either color
        # Edge colors: (0,1)=0, (0,2)=1, (1,2)=0
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions
        assert reward == 1.0, "Should return terminal_reward_success"
        assert done is True, "Episode should be done"
        assert "cliques_lists" in info
        assert "max_clique_size" in info
        # No color should have a 3-clique
        for color_key, cliques in info["cliques_lists"].items():
            if cliques:
                max_clique = max(len(c) for c in cliques)
                assert max_clique < 3, f"{color_key} should not have 3-clique"

    def test_simple_reward_fully_colored_with_violation(self):
        """Test SimpleRewardStrategy when fully colored with violation.

        Expected: reward_loss (-1.0), done=True
        """
        # Setup: 3-vertex complete graph, all same color (forms triangle)
        obs = torch.tensor([0, 0, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions
        assert reward == -1.0, "Should return reward_loss"
        assert done is True, "Episode should be done (fully colored)"
        assert "max_clique_size" in info
        # Color 0 should have a 3-clique
        assert "color_0" in info["max_clique_size"]
        assert info["max_clique_size"]["color_0"] >= 3, "Should detect triangle"

    def test_simple_reward_not_fully_colored(self):
        """Test SimpleRewardStrategy when graph is not fully colored.

        Expected: reward_loss (-1.0), done=False (regardless of violations)
        """
        # Setup: 3-vertex graph with uncolored edges
        obs = torch.tensor([-1, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions
        assert reward == -1.0, "Should return reward_loss"
        assert done is False, "Episode should not be done (has uncolored)"
        assert "cliques_lists" in info

    def test_simple_reward_cumulative_mode(self):
        """Test SimpleRewardStrategy in cumulative mode.

        Expected: rewards accumulate across steps
        """
        obs_incomplete = torch.tensor([-1, 1, 0], dtype=torch.float32)
        obs_complete = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=[0, 1],
                                        cumulative=True)

        # First step: incomplete, should get -1.0
        reward1, done1, _ = strategy.compute_reward(obs_incomplete)
        assert reward1 == -1.0, "First step should be -1.0"
        assert done1 is False

        # Second step: complete, should get cumulative total
        reward2, done2, _ = strategy.compute_reward(obs_complete)
        assert reward2 == 0.0, "Cumulative should be -1.0 + 1.0 = 0.0"
        assert done2 is True


class TestColorSumRewardStrategy:
    """Test suite for ColorSumRewardStrategy."""

    def test_colorsum_reward_per_color_contributions(self):
        """Test ColorSumRewardStrategy sums rewards across colors.

        Expected: reward = sum of individual color rewards
        """
        # Setup: 3-vertex graph fully colored
        # (0,1)=0, (0,2)=1, (1,2)=0 - no triangles in either color
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(
            max_clique_sizes=[3, 3],  # Both colors target size 3
            reward_loss=0.0,
            reward_success=1.0,
            reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions: both colors should contribute reward_success
        assert reward == 2.0, "Sum rewards from both colors (1.0 + 1.0)"
        assert done is True, "Episode should be done (fully colored)"
        assert info["is_counterexample"] is True, "Should be counterexample"

    def test_colorsum_reward_counterexample_detection(self):
        """Test ColorSumRewardStrategy detects counterexamples."""
        # Setup: 4-vertex graph, fully colored, no cliques of size 3
        # Using a specific coloring that avoids triangles
        obs = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(max_clique_sizes=[3, 3],
                                          reward_loss=0.0,
                                          reward_success=1.0,
                                          reward_colors=[0, 1])

        _, done, info = strategy.compute_reward(obs)

        # Assertions
        assert done is True, "Should be done (fully colored)"
        assert "is_counterexample" in info
        assert info["is_counterexample"] is True, "Should detect counterexample"

    def test_colorsum_reward_no_counterexample_with_violation(self):
        """Test ColorSumRewardStrategy when violations exist.

        Expected: info["is_counterexample"]=False when cliques violate sizes
        """
        # Setup: 3-vertex triangle in color 0
        obs = torch.tensor([0, 0, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(max_clique_sizes=[3, 3],
                                          reward_loss=0.0,
                                          reward_success=1.0,
                                          reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions
        assert done is True, "Should be done (fully colored)"
        assert reward == 1.0, "Only color 1 should contribute success"
        assert info["is_counterexample"] is False, "Should NOT be \
            counterexample"

    def test_colorsum_reward_not_fully_colored(self):
        """Test ColorSumRewardStrategy when graph is not fully colored.

        Expected: done=False, no counterexample detection.
        """
        # Setup: partially colored graph
        obs = torch.tensor([-1, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(max_clique_sizes=[3, 3],
                                          reward_loss=0.0,
                                          reward_success=1.0,
                                          reward_colors=[0, 1])

        reward, done, info = strategy.compute_reward(obs)

        # Assertions
        assert done is False, "Should not be done (has uncolored edges)"
        assert reward == 0.0, "All colors should return reward_loss"
        assert info["is_counterexample"] is False, "Cannot detect \
            counterexample"

    def test_colorsum_cumulative_mode(self):
        """Test ColorSumRewardStrategy in cumulative mode.

        Expected: total_reward persists across compute_reward calls
        """
        obs_incomplete = torch.tensor([-1, 1, 0], dtype=torch.float32)
        obs_complete = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(max_clique_sizes=[3, 3],
                                          reward_loss=-0.01,
                                          reward_success=1.0,
                                          reward_colors=[0, 1],
                                          cumulative=True)

        # First step: incomplete
        reward1, done1, _ = strategy.compute_reward(obs_incomplete)
        assert reward1 == -0.02, "First step should be -0.02 (2 * reward_loss)"
        assert done1 is False

        # Second step: complete, should accumulate
        reward2, done2, _ = strategy.compute_reward(obs_complete)
        assert reward2 == 1.98, "Cumulative should be -0.02 + 2.0 = 1.98"
        assert done2 is True


class TestRewardStrategyErrorHandling:
    """Test error handling and edge cases for reward strategies."""

    def test_reward_strategy_with_none_colors(self):
        """Test reward strategies handle None reward_colors correctly.

        Expected: Should raise error or handle gracefully
        Note: This tests the current behavior - TypeError occurs when
        attempting to iterate over None in compute_reward. This is
        incidental behavior, not explicit validation.
        """
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=None)

        # This should raise an error when compute_reward is called
        with pytest.raises(TypeError):
            strategy.compute_reward(obs)

    def test_reward_strategy_with_invalid_int_colors(self):
        """Test single int for reward_colors raises appropriate error.

        Expected: TypeError when iterating over int
        Note: This tests the current behavior - TypeError occurs when
        attempting to iterate over an int in compute_reward. This is
        incidental behavior, not explicit validation.
        """
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(
            max_clique_size=3,
            reward_loss=-1.0,
            terminal_reward_success=1.0,
            reward_colors=2  # Invalid: int instead of list
        )

        # Should raise TypeError when trying to iterate
        with pytest.raises(TypeError):
            strategy.compute_reward(obs)

    def test_colorsum_strategy_with_none_colors(self):
        """Test ColorSumRewardStrategy handles None reward_colors correctly.

        Expected: TypeError when iterating over None
        Note: This tests the current behavior - TypeError occurs when
        attempting to iterate over None in compute_reward. This is
        incidental behavior, not explicit validation.
        """
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(max_clique_sizes=[3, 3],
                                          reward_loss=-1.0,
                                          reward_success=1.0,
                                          reward_colors=None)

        # Should raise TypeError when trying to iterate
        with pytest.raises(TypeError):
            strategy.compute_reward(obs)

    def test_colorsum_strategy_with_invalid_int_colors(self):
        """Test ColorSumRewardStrategy with int reward_colors raises error.

        Expected: TypeError when iterating over int
        Note: This tests the current behavior - TypeError occurs when
        attempting to iterate over an int in compute_reward. This is
        incidental behavior, not explicit validation.
        """
        obs = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = ColorSumRewardStrategy(
            max_clique_sizes=[3, 3],
            reward_loss=-1.0,
            reward_success=1.0,
            reward_colors=2  # Invalid: int instead of list
        )

        # Should raise TypeError when trying to iterate
        with pytest.raises(TypeError):
            strategy.compute_reward(obs)

    def test_reward_strategy_reset_info(self):
        """Test that reward info is properly reset between episodes."""
        obs1 = torch.tensor([0, 0, 0], dtype=torch.float32)
        obs2 = torch.tensor([0, 1, 0], dtype=torch.float32)

        strategy = SimpleRewardStrategy(max_clique_size=3,
                                        reward_loss=-1.0,
                                        terminal_reward_success=1.0,
                                        reward_colors=[0, 1])

        # First compute with violations
        _, _, info1 = strategy.compute_reward(obs1)
        assert "color_0" in info1["max_clique_size"]

        # Reset and compute again
        strategy.reset_reward_info()
        _, _, info2 = strategy.compute_reward(obs2)

        # Info should be fresh (different from previous)
        assert info2 is not info1, "Info should be new object"
