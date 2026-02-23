"""Tests for Gymnasium wrapper environments."""

import gymnasium
import pytest
import torch
from unittest.mock import patch

from ramsey.gym_ramsey_env import (
    BaseRamseyGymEnv,
    RamseyGymEnvV0,
    RamseyGymEnvV1,
    RamseyGymEnvV2,
)
from ramsey.action_types import (
    DefaultActionStrategy,
    TwoActionStrategy,
    CirculantActionStrategy,
)
from ramsey.rewards import SimpleRewardStrategy


class DummyBaseRamseyGymEnv(BaseRamseyGymEnv):
    """A minimal concrete implementation of BaseRamseyGymEnv for testing."""

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(1)  # Minimal action space for testing

    def truncate_episode(self) -> bool:
        return False  # Default: no truncation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestBaseWrapperObservationSpace:
    """Test suite for BaseRamseyGymEnv observation space definition."""

    def test_base_wrapper_observation_space_definition(self):
        """Test that observation_space is Box with correct bounds and dtype.

        Verifies:
        - Space is gymnasium.spaces.Box
        - Low bound is 0
        - High bound is n_colors - 1
        - Shape is (n_edges,)
        - dtype is int
        """
        env = DummyBaseRamseyGymEnv(
            n_vertices=4,
            clique_sizes=[3, 3],
            init_method_name="empty",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        # Verify observation space properties
        assert hasattr(env, "observation_space"), \
            "Environment should have observation_space attribute"

        obs_space = env.observation_space
        assert isinstance(obs_space, gymnasium.spaces.Box), \
            "Observation space should be Box"

        # For n=4: C(4,2) = 6 edges
        expected_shape = (6,)
        assert obs_space.shape == expected_shape, \
            f"Observation space shape should be {expected_shape}"

        # n_colors = 2, so high should be 1
        assert obs_space.low.min() == 0, "Low bound should be 0"
        assert obs_space.high.max() == 1, \
            "High bound should be n_colors - 1 = 1"

        # dtype should be int
        assert obs_space.dtype == int, "dtype should be int"

    def test_base_wrapper_obs_bounds_mismatch_documented(self):
        """Test observation space bounds don't match uncolored observations.

        KNOWN GOTCHA: The wrapper defines Box(low=0, ...) but uncolored
        observations contain -1 values, causing observation_space.contains(obs)
        to return False.

        Verifies:
        - Observation space declares low=0
        - Uncolored observations contain -1
        - observation_space.contains(obs) returns False (API compliance gotcha)
        """
        env = DummyBaseRamseyGymEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        obs, _ = env.reset()

        # Verify observation contains -1 values
        assert (obs == -1).any().item(), \
            "Uncolored observation should contain -1 values"

        # KNOWN GOTCHA: observation_space.contains() returns False
        # because bounds declare low=0 but observation contains -1
        assert not env.observation_space.contains(obs.numpy()), \
            "observation_space.contains(obs) should return False " \
            "(documents known bounds mismatch)"


class TestBaseWrapperReset:
    """Test suite for BaseRamseyGymEnv reset functionality."""

    def test_base_wrapper_reset_returns_tuple(self):
        """Test that reset() returns (obs, info) in Gymnasium format.

        Verifies:
        - Returns tuple of length 2
        - First element is torch.Tensor observation
        - Second element is dict info
        """
        env = DummyBaseRamseyGymEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        result = env.reset()

        assert isinstance(result, tuple), \
            "reset() should return a tuple"
        assert len(result) == 2, \
            "reset() should return tuple of length 2"

        obs, info = result
        assert isinstance(obs, torch.Tensor), \
            "First element should be torch.Tensor observation"
        assert isinstance(info, dict), \
            "Second element should be dict info"

        # Verify observation has correct shape
        expected_shape = (3,)  # n=3 => C(3,2) = 3 edges
        assert obs.shape == expected_shape, \
            f"Observation shape should be {expected_shape}"


class TestBaseWrapperStep:
    """Test suite for BaseRamseyGymEnv step functionality."""

    def test_base_wrapper_step_return_order(self):
        """Test step() return: (obs, reward, terminated, truncated, info).

        Follows standard Gymnasium convention:
        (obs, reward, terminated, truncated, info)

        Verifies:
        - Returns tuple of length 5
        - Element order matches Gymnasium convention
        - Types are correct: Tensor, float, bool, bool, dict
        """
        env = DummyBaseRamseyGymEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
            action_strategy=DefaultActionStrategy(),
        )

        env.reset()
        result = env.step(0)

        assert isinstance(result, tuple), \
            "step() should return a tuple"
        assert len(result) == 5, \
            "step() should return tuple of length 5"

        obs, reward, terminated, truncated, info = result

        # Verify types
        assert isinstance(obs, torch.Tensor), \
            "Element 0 should be torch.Tensor observation"
        assert isinstance(reward, (int, float)), \
            "Element 1 should be numeric reward"
        assert isinstance(terminated, bool), \
            "Element 2 should be bool terminated (done)"
        assert isinstance(truncated, bool), \
            "Element 3 should be bool truncated"
        assert isinstance(info, dict), \
            "Element 4 should be dict info"


class TestBaseWrapperTrajectoryTracking:
    """Test suite for trajectory tracking in BaseRamseyGymEnv."""

    def test_base_wrapper_trajectory_tracking(self):
        """Test that trajectory resets on reset() and accumulates on step().

        Verifies:
        - Trajectory is empty after reset()
        - Trajectory accumulates observations, actions, rewards after steps
        - Trajectory contains correct number of entries
        """
        env = DummyBaseRamseyGymEnv(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
            action_strategy=DefaultActionStrategy(),
        )

        # After initialization, trajectory should exist
        assert hasattr(env, "trajectory"), \
            "Environment should have trajectory attribute"

        # After reset, trajectory should be empty
        env.reset()
        assert len(env.trajectory.observations) == 0, \
            "Trajectory observations should be empty after reset"
        assert len(env.trajectory.actions) == 0, \
            "Trajectory actions should be empty after reset"
        assert len(env.trajectory.rewards) == 0, \
            "Trajectory rewards should be empty after reset"

        # Take first step
        obs1, reward1, _, _, info1 = env.step(0)
        assert len(env.trajectory.observations) == 1, \
            "Trajectory should have 1 observation after 1 step"
        assert len(env.trajectory.actions) == 1, \
            "Trajectory should have 1 action after 1 step"
        assert len(env.trajectory.rewards) == 1, \
            "Trajectory should have 1 reward after 1 step"

        # Verify stored values
        assert torch.equal(env.trajectory.observations[0], obs1), \
            "Stored observation should match returned observation"
        assert env.trajectory.actions[0] == 0, \
            "Stored action should be 0"
        assert env.trajectory.rewards[0] == reward1, \
            "Stored reward should match returned reward"
        assert env.trajectory.info == info1, \
            "Stored info should match returned info"

        # Take second step
        _, _, _, _, _ = env.step(1)
        assert len(env.trajectory.observations) == 2, \
            "Trajectory should have 2 observations after 2 steps"
        assert len(env.trajectory.actions) == 2, \
            "Trajectory should have 2 actions after 2 steps"
        assert len(env.trajectory.rewards) == 2, \
            "Trajectory should have 2 rewards after 2 steps"

        # Reset again and verify trajectory is cleared
        env.reset()
        assert len(env.trajectory.observations) == 0, \
            "Trajectory should be cleared after second reset"


class TestV0ActionSpaceAndTruncation:
    """Test suite for RamseyGymEnvV0 action space and truncation logic."""

    def test_v0_action_space_and_truncation(self):
        """Test V0 uses DefaultActionStrategy and Discrete action space.

        Verifies:
        - Action space is Discrete(n_edges * n_colors)
        - Truncation occurs at n_edges * n_colors steps
        - Uses DefaultActionStrategy for action decoding
        """
        n_vertices = 4
        n_edges = 6  # C(4,2)
        n_colors = 2

        env = RamseyGymEnvV0(
            n_vertices=n_vertices,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        # Verify action space
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), \
            "Action space should be Discrete"

        expected_action_dim = n_edges * n_colors
        assert env.action_space.n == expected_action_dim, \
            f"Action space should have {expected_action_dim} actions"

        # Verify action strategy
        assert isinstance(env.env.action_strategy, DefaultActionStrategy), \
            "Should use DefaultActionStrategy"

        # Test truncation logic
        env.reset()

        # Take steps up to max_steps - 1
        max_steps = n_edges * n_colors
        truncated = False
        terminated = False
        for i in range(max_steps - 1):
            if terminated:
                break
            action = i % expected_action_dim
            _, _, terminated, truncated, _ = env.step(action)
            if not terminated:
                assert truncated is False, \
                    f"Should not truncate before step {max_steps}"

        # If episode terminated naturally, can't test truncation
        # Otherwise, take one more step to reach max_steps
        if not terminated:
            _, _, terminated, truncated, _ = env.step(0)
            if not terminated:
                assert truncated is True, \
                    f"Should truncate at step {max_steps}"
            # If terminated and truncated, that's valid too
            # (episode ended at exactly max_steps)


class TestV1ActionSpaceAndBehavior:
    """Test suite for RamseyGymEnvV1 action space and behavioral quirks."""

    def test_v1_action_space_and_behavior(self):
        """Test V1 uses TwoActionStrategy with modified observation space.

        Verifies:
        - Uses TwoActionStrategy
        - Action space is Discrete(2)
        - Observation space shape is (n_colors + n_edges,) not (n_edges,)
        - No truncation (truncate_episode returns False by default)

        Behavioral quirk: V1 changes observation space to include one-hot
        encoding of current vertex index, making it (n_colors + n_edges,).
        """
        n_vertices = 4
        n_edges = 6  # C(4,2)
        n_colors = 2

        env = RamseyGymEnvV1(
            n_vertices=n_vertices,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        # Verify action space
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), \
            "Action space should be Discrete"
        assert env.action_space.n == 2, \
            "Action space should have 2 actions"

        # Verify action strategy
        assert isinstance(env.env.action_strategy, TwoActionStrategy), \
            "Should use TwoActionStrategy"

        # Verify modified observation space
        expected_obs_shape = (n_colors + n_edges,)
        assert env.observation_space.shape == expected_obs_shape, \
            f"V1 observation space should be {expected_obs_shape}"

        # Verify no truncation by default
        env.reset()

        # Take many steps - should never truncate
        for i in range(20):
            _, _, terminated, truncated, _ = env.step(i % 2)
            if terminated:
                break
            assert truncated is False, \
                "V1 should not truncate by default"


class TestV2CirculantActionAndTruncation:
    """Test suite for RamseyGymEnvV2 circulant action strategy."""

    def test_v2_circulant_action_and_truncation(self):
        """Test V2 uses CirculantActionStrategy with multi-edge updates.

        Verifies:
        - Uses CirculantActionStrategy
        - Action space is Discrete(n_chord_lengths * n_colors)
        - n_chord_lengths = n_vertices // 2
        - Truncates at n_chord_lengths * n_colors steps
        - Single action can update multiple edges (chord length behavior)
        """
        n_vertices = 6
        n_chord_lengths = n_vertices // 2  # = 3
        n_colors = 2

        env = RamseyGymEnvV2(
            n_vertices=n_vertices,
            clique_sizes=[3, 3],
            init_method_name="uncolored",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
        )

        # Verify action space
        assert isinstance(env.action_space, gymnasium.spaces.Discrete), \
            "Action space should be Discrete"

        expected_action_dim = n_chord_lengths * n_colors
        assert env.action_space.n == expected_action_dim, \
            f"Action space should have {expected_action_dim} actions"

        # Verify action strategy
        assert isinstance(env.env.action_strategy, CirculantActionStrategy), \
            "Should use CirculantActionStrategy"

        # Verify n_chord_lengths
        assert env.env.n_chord_lengths == n_chord_lengths, \
            f"n_chord_lengths should be {n_chord_lengths}"

        # Test truncation logic
        env.reset()

        max_steps = n_chord_lengths * n_colors
        truncated = False
        terminated = False
        for i in range(max_steps - 1):
            if terminated:
                break
            action = i % expected_action_dim
            _, _, terminated, truncated, _ = env.step(action)
            if not terminated:
                assert truncated is False, \
                    f"Should not truncate before step {max_steps}"

        # If episode terminated naturally, can't test truncation
        # Otherwise, take one more step to reach max_steps
        if not terminated:
            _, _, terminated, truncated, _ = env.step(0)
            if not terminated:
                assert truncated is True, \
                    f"Should truncate at step {max_steps}"
            # If terminated and truncated, that's valid too

        # Test multi-edge update behavior
        env.reset()
        obs_before, _ = env.reset()

        # Count uncolored edges before action
        uncolored_before = (obs_before == -1).sum().item()

        # Take action 0 (chord_length=1, color=0)
        obs_after, _, _, _, _ = env.step(0)

        # Count uncolored edges after action
        uncolored_after = (obs_after == -1).sum().item()

        # Circulant action should color multiple edges (n_vertices edges)
        edges_colored = uncolored_before - uncolored_after
        assert edges_colored == n_vertices, \
            f"Circulant action should color {n_vertices} edges, " \
            f"colored {edges_colored}"


class TestWrapperRenderModes:
    """Test suite for render mode support across wrappers."""

    def test_wrapper_render_modes(self):
        """Test render() behavior for different modes.

        Verifies:
        - "None" mode does nothing (returns None)
        - "static" mode calls rendering.static_render (or raises
            NotImplementedError)
        - "animated" mode raises NotImplementedError
        - Invalid modes should fail during initialization

        Uses mocking to avoid actual matplotlib/networkx rendering.
        """
        env = RamseyGymEnvV0(
            n_vertices=3,
            clique_sizes=[3, 3],
            init_method_name="empty",
            reward_strategy=SimpleRewardStrategy(
                max_clique_size=3,
                reward_loss=-1.0,
                terminal_reward_success=1.0,
                reward_colors=[0, 1],
            ),
            render_mode="static",
        )

        # Test "None" mode - should do nothing
        # pylint: disable=assignment-from-none
        result = env.render(mode="None")
        assert result is None, "None mode should return None"

        # Test "static" mode - mock the rendering call
        with patch("ramsey.rendering.static_render") as mock_render:
            env.render(mode="static")
            mock_render.assert_called_once_with(env.env)

        # Test "animated" mode - should raise NotImplementedError
        with pytest.raises(NotImplementedError,
                           match="Render mode 'animated' is not implemented"):
            env.render(mode="animated")

        # Test invalid mode during initialization
        with pytest.raises(AssertionError):
            RamseyGymEnvV0(
                n_vertices=3,
                clique_sizes=[3, 3],
                init_method_name="empty",
                reward_strategy=SimpleRewardStrategy(
                    max_clique_size=3,
                    reward_loss=-1.0,
                    terminal_reward_success=1.0,
                    reward_colors=[0, 1],
                ),
                render_mode="invalid_mode",
            )
