"""Tests for RamseyEnv core environment functionality."""

import pytest
import torch

from ramsey.ramsey_env import RamseyEnv
from ramsey.action_types import DefaultActionStrategy
from ramsey.rewards import SimpleRewardStrategy


class TestRamseyEnvInitialization:
    """Test suite for RamseyEnv initialization and validation."""

    def test_env_init_validates_n_vertices(self):
        """Test that n_vertices < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid number of vertices"):
            RamseyEnv(n_vertices=0,
                      clique_sizes=[3, 3],
                      init_method_name="empty")

        with pytest.raises(ValueError, match="Invalid number of vertices"):
            RamseyEnv(n_vertices=-5,
                      clique_sizes=[3, 3],
                      init_method_name="empty")

    def test_env_init_validates_init_method(self):
        """Test that unknown init_method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown init_method"):
            RamseyEnv(n_vertices=5,
                      clique_sizes=[3, 3],
                      init_method_name="invalid_method")

        with pytest.raises(ValueError, match="Unknown init_method"):
            RamseyEnv(n_vertices=5,
                      clique_sizes=[3, 3],
                      init_method_name="nonexistent")

    def test_env_init_accepts_valid_parameters(self):
        """Test that valid parameters initialize successfully."""
        env = RamseyEnv(n_vertices=5,
                        clique_sizes=[3, 3],
                        init_method_name="empty")

        assert env.n_vertices == 5
        assert env.clique_sizes == [3, 3]
        assert env.init_method_name == "empty"
        assert env.n_colors == 2
        assert env.n_edges == 10  # C(5,2) = 10


class TestRamseyEnvReset:
    """Test suite for RamseyEnv reset functionality."""

    def test_env_reset_initializes_state_correctly(self):
        """Test that reset() sets steps=0, done=False, reward=0.0."""
        env = RamseyEnv(n_vertices=4,
                        clique_sizes=[3, 3],
                        init_method_name="empty")

        obs, info = env.reset()

        assert env.steps == 0, "Steps should be initialized to 0"
        assert env.done is False, "Done should be initialized to False"
        assert env.reward == 0.0, "Reward should be initialized to 0.0"
        assert isinstance(obs, torch.Tensor), "Observation should be a tensor"
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_env_reset_applies_init_function_empty(self):
        """Test that reset() with 'empty' initializes with zeros."""
        env = RamseyEnv(n_vertices=4,
                        clique_sizes=[3, 3],
                        init_method_name="empty")

        obs, _ = env.reset()

        expected_size = 6  # C(4,2) = 6 edges
        assert obs.shape[0] == expected_size, \
            "Should have correct number of edges"
        assert torch.all(obs == 0), \
            "All edges should be colored 0 (empty)"

    def test_env_reset_applies_init_function_uncolored(self):
        """Test that reset() with 'uncolored' initializes with -1."""
        env = RamseyEnv(n_vertices=4,
                        clique_sizes=[3, 3],
                        init_method_name="uncolored")

        obs, _ = env.reset()

        expected_size = 6  # C(4,2) = 6 edges
        assert obs.shape[0] == expected_size, \
            "Should have correct number of edges"
        assert torch.all(obs == -1), \
            "All edges should be uncolored (-1)"

    def test_env_reset_returns_correct_device(self):
        """Test that reset() returns observation on correct device."""
        device = torch.device("cpu")
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="empty",
                        device=device)

        obs, _ = env.reset()

        assert obs.device == device, "Observation should be on specified device"


class TestRamseyEnvStep:
    """Test suite for RamseyEnv step functionality."""

    def test_env_step_increments_steps(self):
        """Test that step() increments the step counter."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="uncolored",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        env.reset()
        assert env.steps == 0, "Steps should start at 0"

        env.step(0)
        assert env.steps == 1, "Steps should increment to 1 after first step"

        env.step(1)
        assert env.steps == 2, "Steps should increment to 2 after second step"

    def test_env_step_mutates_adjacency_vec(self):
        """Test that step() correctly changes edge colors via actions."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="uncolored",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        obs, _ = env.reset()
        # All edges should be -1 (uncolored)
        assert torch.all(obs == -1), \
            "Initial state should be all uncolored"

        # Action 0: color edge 0 with color 0
        obs, _, _, _ = env.step(0)
        assert obs[0] == 0, "Edge 0 should be colored 0"
        assert obs[1] == -1, "Edge 1 should remain uncolored"

        # Action 4: color edge 1 with color 1
        # (action // n_edges = 1, action % n_edges = 1)
        obs, _, _, _ = env.step(4)
        assert obs[1] == 1, "Edge 1 should be colored 1"

    def test_env_step_after_done_raises_error(self):
        """Test that stepping after done=True raises RuntimeError."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="empty",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        env.reset()
        # Manually set done to True to simulate terminal state
        env.done = True

        with pytest.raises(RuntimeError,
                           match="Episode has finished. Call reset"):
            env.step(0)

    def test_env_step_with_missing_action_strategy_raises(self):
        """Test that step() raises error when action_strategy is None."""
        env = RamseyEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="empty",
            action_strategy=None,  # Missing strategy
            reward_strategy=SimpleRewardStrategy(max_clique_size=3,
                                                 reward_loss=-1.0,
                                                 terminal_reward_success=1.0,
                                                 reward_colors=[0, 1]))

        env.reset()

        # AttributeError is incidental - raised when None.decode_action() is
        # called
        with pytest.raises(AttributeError, match="decode_action"):
            env.step(0)

    def test_env_step_with_missing_reward_strategy_raises(self):
        """Test that step() raises error when reward_strategy is None."""
        env = RamseyEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="empty",
            action_strategy=DefaultActionStrategy(),
            reward_strategy=None  # Missing strategy
        )

        env.reset()

        # AttributeError is incidental - raised when None.compute_reward() is
        # called
        with pytest.raises(AttributeError, match="compute_reward"):
            env.step(0)

    def test_env_step_returns_correct_tuple(self):
        """Test that step() returns (obs, reward, done, info) tuple."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="uncolored",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        env.reset()
        obs, reward, done, info = env.step(0)

        assert isinstance(obs, torch.Tensor), "Observation should be a tensor"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_env_step_before_reset_raises_error(self):
        """Test that calling step() before reset() raises AttributeError."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="empty",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        # Call step before reset - should raise AttributeError for missing
        # 'steps' attribute (steps is the first attribute accessed in step(),
        # before adjacency_vec)
        with pytest.raises(AttributeError, match="steps"):
            env.step(0)


class TestRamseyEnvIntegration:
    """Integration tests for complete RamseyEnv episodes."""

    def test_env_full_episode_reaches_done(self):
        """Test that a full episode reaches done=True deterministically.
        
        Uses a small graph (n=3, 3 edges) with SimpleRewardStrategy.
        Color all edges systematically to trigger terminal condition.
        """
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="uncolored",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        env.reset()

        # With n=3, we have 3 edges to color Color all edges: edge 0 with color
        # 0, edge 1 with color 1, edge 2 with color 0 This creates a
        # configuration that should reach done=True
        _, _, done, _ = env.step(0)  # Color edge 0 with color 0 (action=0)
        assert not done, "Should not be done after 1 edge"

        _, _, done, _ = env.step(
            4)  # Color edge 1 with color 1 (action=n_edges*1+1)
        assert not done, "Should not be done after 2 edges"

        _, _, done, _ = env.step(2)  # Color edge 2 with color 0 (action=2)

        # After coloring all edges, the episode must be done
        assert done is True, "Episode must reach done=True after coloring \
            all edges in small graph"

    def test_env_reset_after_done(self):
        """Test that reset() can be called after done=True."""
        env = RamseyEnv(n_vertices=3,
                        clique_sizes=[3, 3],
                        init_method_name="empty",
                        action_strategy=DefaultActionStrategy(),
                        reward_strategy=SimpleRewardStrategy(
                            max_clique_size=3,
                            reward_loss=-1.0,
                            terminal_reward_success=1.0,
                            reward_colors=[0, 1]))

        # First episode
        env.reset()
        env.done = True  # Simulate episode end

        # Reset should work
        _, _ = env.reset()
        assert env.done is False, "Done should be reset to False"
        assert env.steps == 0, "Steps should be reset to 0"
        assert env.reward == 0.0, "Reward should be reset to 0.0"
