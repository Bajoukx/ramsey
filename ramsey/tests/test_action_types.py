"""Tests for ramsey.action_types module.

This module tests the action decoding strategies that convert integer actions
into (color, edge_index) or (color, edge_indices_list) tuples for the
environment.
"""

# pylint: disable=protected-access

import pytest
import itertools

import torch

from ramsey import action_types


class MockEnv:
    """Mock environment for testing action strategies."""

    def __init__(self,
                 n_vertices=4,
                 n_edges=6,
                 n_colors=2,
                 n_chord_lengths=None,
                 adjacency_vec=None):
        self.n_vertices = n_vertices
        self.n_edges = n_edges
        self.n_colors = n_colors
        if n_chord_lengths is not None:
            self.n_chord_lengths = n_chord_lengths
        else:
            self.n_chord_lengths = n_vertices // 2

        # Generate edges in combinations order
        self.all_edges = list(itertools.combinations(range(n_vertices), 2))

        # For TwoActionStrategy: env.env.adjacency_vec structure
        # Create nested env structure to match the actual usage
        self.env = self

        # Set adjacency_vec with vertex one-hot at the end
        if adjacency_vec is not None:
            self.adjacency_vec = adjacency_vec
        else:
            # Default: create adjacency vec with one-hot at end
            # Last n_edges elements are one-hot encoding of current vertex
            self.adjacency_vec = torch.zeros(n_edges * 2)  # Simple default


class TestDefaultActionStrategy:
    """Tests for DefaultActionStrategy."""

    def test_basic_action_mapping(self):
        """Test basic action to (color, edge_idx) mapping."""
        strategy = action_types.DefaultActionStrategy()
        env = MockEnv(n_vertices=3, n_edges=3, n_colors=2)

        # Action space size = n_edges * n_colors = 3 * 2 = 6
        # Actions 0-2: color 0, edges 0-2
        # Actions 3-5: color 1, edges 0-2

        color, edge_idx = strategy.decode_action(env, action=0)
        assert color == 0 and edge_idx == 0, \
            f"Action 0 should map to (0, 0), got ({color}, {edge_idx})"

        color, edge_idx = strategy.decode_action(env, action=2)
        assert color == 0 and edge_idx == 2, \
            f"Action 2 should map to (0, 2), got ({color}, {edge_idx})"

        color, edge_idx = strategy.decode_action(env, action=3)
        assert color == 1 and edge_idx == 0, \
            f"Action 3 should map to (1, 0), got ({color}, {edge_idx})"

        color, edge_idx = strategy.decode_action(env, action=5)
        assert color == 1 and edge_idx == 2, \
            f"Action 5 should map to (1, 2), got ({color}, {edge_idx})"

    def test_action_mapping_with_larger_graph(self):
        """Test action mapping with n_vertices=4 (6 edges)."""
        strategy = action_types.DefaultActionStrategy()
        env = MockEnv(n_vertices=4, n_edges=6, n_colors=2)

        # Action space size = 6 * 2 = 12
        # Actions 0-5: color 0, edges 0-5
        # Actions 6-11: color 1, edges 0-5

        color, edge_idx = strategy.decode_action(env, action=0)
        assert (color, edge_idx) == (0, 0)

        color, edge_idx = strategy.decode_action(env, action=5)
        assert (color, edge_idx) == (0, 5)

        color, edge_idx = strategy.decode_action(env, action=6)
        assert (color, edge_idx) == (1, 0)

        color, edge_idx = strategy.decode_action(env, action=11)
        assert (color, edge_idx) == (1, 5)

    def test_action_mapping_with_three_colors(self):
        """Test action mapping with 3 colors."""
        strategy = action_types.DefaultActionStrategy()
        env = MockEnv(n_vertices=3, n_edges=3, n_colors=3)

        # Action space size = 3 * 3 = 9
        # Actions 0-2: color 0
        # Actions 3-5: color 1
        # Actions 6-8: color 2

        color, edge_idx = strategy.decode_action(env, action=0)
        assert (color, edge_idx) == (0, 0)

        color, edge_idx = strategy.decode_action(env, action=3)
        assert (color, edge_idx) == (1, 0)

        color, edge_idx = strategy.decode_action(env, action=6)
        assert (color, edge_idx) == (2, 0)

        color, edge_idx = strategy.decode_action(env, action=8)
        assert (color, edge_idx) == (2, 2)

    def test_boundary_action_at_max(self):
        """Test action at boundary (max valid action)."""
        strategy = action_types.DefaultActionStrategy()
        env = MockEnv(n_vertices=3, n_edges=3, n_colors=2)

        # Max valid action = n_edges * n_colors - 1 = 5
        max_action = env.n_edges * env.n_colors - 1
        color, edge_idx = strategy.decode_action(env, action=max_action)

        expected_color = max_action // env.n_edges
        expected_edge = max_action % env.n_edges

        assert (color, edge_idx) == (expected_color, expected_edge), \
            f"Max action {max_action} should map to ({expected_color}, \
                {expected_edge})"

    def test_out_of_range_action_produces_invalid_color(self):
        """Test that out-of-range actions produce invalid color values.
        
        TODO: Implement action ranges error handling.
        """
        strategy = action_types.DefaultActionStrategy()
        env = MockEnv(n_vertices=3, n_edges=3, n_colors=2)

        # Invalid action beyond action space
        invalid_action = env.n_edges * env.n_colors  # = 6
        color, _ = strategy.decode_action(env, invalid_action)

        # This should produce color = 2 (which is invalid for n_colors=2)
        assert color >= env.n_colors, \
            f"Out-of-range action should produce invalid color, got {color}"


class TestTwoActionStrategy:
    """Tests for TwoActionStrategy.
    
    This strategy uses an n_colors action space where the action directly
    represents the color to apply, and the edge index is always inferred from
    the observation (vertex one-hot encoding in the last n_edges elements).
    """

    def test_action_equals_color_directly(self):
        """Test that action directly maps to color."""
        strategy = action_types.TwoActionStrategy()
        env = MockEnv(n_vertices=3,
                      n_edges=3,
                      n_colors=2,
                      adjacency_vec=torch.zeros(4))

        # Action 0 should produce color 0
        color, _ = strategy.decode_action(env, action=0)
        assert color == 0, f"Action 0 should be color 0, got {color}"

        # Action 1 should produce color 1
        color, _ = strategy.decode_action(env, action=1)
        assert color == 1, f"Action 1 should be color 1, got {color}"

    def test_edge_always_inferred_from_observation(self):
        """Test that edge_idx is always inferred from observation."""
        strategy = action_types.TwoActionStrategy()

        # Create env with vertex one-hot at position 2
        one_hot = torch.zeros(3)
        one_hot[2] = 1.0
        adjacency_vec = torch.cat([torch.zeros(3), one_hot])
        env = MockEnv(n_vertices=3,
                      n_edges=3,
                      n_colors=2,
                      adjacency_vec=adjacency_vec)

        # All actions should infer edge index from one-hot (should be 2)
        for action in range(2):  # n_colors = 2
            _, edge_idx = strategy.decode_action(env, action)
            assert edge_idx == 2, \
                f"Action {action} should infer edge_idx=2, got {edge_idx}"

    def test_infer_edge_idx_from_one_hot_encoding(self):
        """Test _infer_edge_idx extracts the correct index from one-hot."""
        strategy = action_types.TwoActionStrategy()

        # Test with different one-hot positions
        for expected_idx in range(4):
            one_hot = torch.zeros(4)
            one_hot[expected_idx] = 1.0
            adjacency_vec = torch.cat([torch.zeros(4), one_hot])
            env = MockEnv(n_vertices=4,
                          n_edges=4,
                          n_colors=2,
                          adjacency_vec=adjacency_vec)

            inferred_idx = strategy._infer_edge_idx(env)
            assert inferred_idx == expected_idx, \
                f"Expected index {expected_idx}, got {inferred_idx}"

    def test_action_space_size_equals_n_colors(self):
        """Test that action space size is n_colors."""
        strategy = action_types.TwoActionStrategy()
        one_hot = torch.zeros(4)
        one_hot[0] = 1.0
        adjacency_vec = torch.cat([torch.zeros(4), one_hot])

        # Test with different n_colors
        for n_colors in [2, 3, 4, 5]:
            env = MockEnv(n_vertices=4,
                          n_edges=4,
                          n_colors=n_colors,
                          adjacency_vec=adjacency_vec)

            # Action space should be n_colors
            # Test all valid actions
            for action in range(n_colors):
                color, edge_idx = strategy.decode_action(env, action)
                assert color == action, f"Action {action} should produce color \
                    {action}, got {color}"
                assert edge_idx == 0, f"Should infer edge_idx=0 from one-hot, \
                    got {edge_idx}"

    def test_different_edge_positions(self):
        """Test different one-hot positions produce different edge indices."""
        strategy = action_types.TwoActionStrategy()

        # Test various edge positions
        for edge_pos in [0, 1, 2]:
            one_hot = torch.zeros(3)
            one_hot[edge_pos] = 1.0
            adjacency_vec = torch.cat([torch.zeros(3), one_hot])
            env = MockEnv(n_vertices=3,
                          n_edges=3,
                          n_colors=2,
                          adjacency_vec=adjacency_vec)

            # All actions should return the same edge position
            for action in range(2):
                _, edge_idx = strategy.decode_action(env, action)
                assert edge_idx == edge_pos, f"With one-hot at {edge_pos}, \
                    should infer edge_idx={edge_pos}, got {edge_idx}"

    def test_three_colors(self):
        """Test with 3 colors to verify action space scaling."""
        strategy = action_types.TwoActionStrategy()
        one_hot = torch.zeros(3)
        one_hot[1] = 1.0
        adjacency_vec = torch.cat([torch.zeros(3), one_hot])
        env = MockEnv(n_vertices=3,
                      n_edges=3,
                      n_colors=3,
                      adjacency_vec=adjacency_vec)

        # Test all 3 actions map to correct colors
        colors = []
        for action in range(3):
            color, edge_idx = strategy.decode_action(env, action)
            colors.append(color)
            # All should infer same edge
            assert edge_idx == 1, \
                f"Action {action} should infer edge_idx=1, got {edge_idx}"

        assert colors == [0, 1, 2], \
            f"Actions 0-2 should map to colors [0, 1, 2], got {colors}"

    def test_integration_with_different_observations(self):
        """Test integration with different observations."""
        strategy = action_types.TwoActionStrategy()

        # Create environments with different one-hot positions
        observations = []
        for pos in range(6):
            one_hot = torch.zeros(6)
            one_hot[pos] = 1.0
            adjacency_vec = torch.cat([torch.zeros(6), one_hot])
            observations.append((pos, adjacency_vec))

        # Test each observation
        for expected_pos, adjacency_vec in observations:
            env = MockEnv(n_vertices=6,
                          n_edges=6,
                          n_colors=3,
                          adjacency_vec=adjacency_vec)

            # Test with different colors/actions
            for action in range(3):
                color, edge_idx = strategy.decode_action(env, action)
                assert color == action, "Action should equal color"
                assert edge_idx == expected_pos, \
                    f"Should infer edge_idx={expected_pos}, got {edge_idx}"

    def test_max_action_boundary(self):
        """Test maximum valid action (n_colors - 1)."""
        strategy = action_types.TwoActionStrategy()
        one_hot = torch.zeros(3)
        one_hot[0] = 1.0
        adjacency_vec = torch.cat([torch.zeros(3), one_hot])

        for n_colors in [2, 3, 4]:
            env = MockEnv(n_vertices=3,
                          n_edges=3,
                          n_colors=n_colors,
                          adjacency_vec=adjacency_vec)

            # Max action should be n_colors - 1
            max_action = n_colors - 1
            color, edge_idx = strategy.decode_action(env, action=max_action)

            assert color == max_action, f"Max action {max_action} should \
                produce color {max_action}, got {color}"
            assert edge_idx == 0, \
                f"Should infer edge_idx=0, got {edge_idx}"


class TestCirculantActionStrategy:
    """Tests for CirculantActionStrategy."""

    def test_chord_length_to_edge_indices_n4_chord1(self):
        """Test chord length 1 for n=4 vertices."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=4, n_edges=6, n_colors=2)

        # Chord length 1: edges (0,1), (1,2), (2,3), (3,0)
        edge_indices = strategy._chord_length_to_edge_indices(n_vertices=4,
                                                              chord_length=1)

        # Convert to actual edges to verify
        actual_edges = [env.all_edges[idx] for idx in edge_indices]
        expected_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]

        # Order may vary, so compare as sets
        assert set(actual_edges) == set(expected_edges), \
            f"Chord 1 edges: expected {expected_edges}, got {actual_edges}"

    def test_chord_length_to_edge_indices_n5_chord1(self):
        """Test chord length 1 for n=5 vertices (pentagon perimeter)."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=5, n_edges=10, n_colors=2)

        # Chord length 1: edges forming pentagon
        edge_indices = strategy._chord_length_to_edge_indices(n_vertices=5,
                                                              chord_length=1)

        actual_edges = [env.all_edges[idx] for idx in edge_indices]
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]

        assert set(actual_edges) == set(expected_edges), \
            f"Chord 1 for n=5: expected {expected_edges}, got {actual_edges}"

    def test_chord_length_to_edge_indices_n5_chord2(self):
        """Test chord length 2 for n=5 vertices (star diagonals)."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=5, n_edges=10, n_colors=2)

        # Chord length 2: (0,2), (1,3), (2,4), (3,0), (4,1)
        edge_indices = strategy._chord_length_to_edge_indices(n_vertices=5,
                                                              chord_length=2)

        actual_edges = [env.all_edges[idx] for idx in edge_indices]
        expected_edges = [(0, 2), (1, 3), (2, 4), (0, 3), (1, 4)]

        assert set(actual_edges) == set(expected_edges), \
            f"Chord 2 for n=5: expected {expected_edges}, got {actual_edges}"

    def test_even_n_duplicate_indices(self):
        """Test that even n_vertices with chord_length = n/2 produces 
        duplicates.
        
        For even n, chord length n/2 represents diameters (opposite vertices),
        and since the graph is undirected, each diameter appears twice.
        """
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=6, n_edges=15, n_colors=2)

        # Chord length 3 for n=6 (diameters)
        edge_indices = strategy._chord_length_to_edge_indices(n_vertices=6,
                                                              chord_length=3)

        # Check for duplicates
        assert len(edge_indices) != len(set(edge_indices)), \
            "Even n with chord_length = n/2 should produce duplicate indices"

        # The edges should be (0,3), (1,4), (2,5) - 3 unique edges
        actual_edges = [env.all_edges[idx] for idx in edge_indices]
        unique_edges = set(actual_edges)
        assert len(unique_edges) == 3, \
            f"6 vertices with chord 3 should have 3 unique diameter edges, \
                got {len(unique_edges)}"

    def test_decode_action_basic(self):
        """Test basic action decoding."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=4, n_edges=6, n_colors=2, n_chord_lengths=2)

        # Action space = n_chord_lengths * n_colors = 2 * 2 = 4
        # Actions 0-1: color 0, chords 1-2
        # Actions 2-3: color 1, chords 1-2

        color, edge_indices = strategy.decode_action(env, action=0)
        assert color == 0, f"Action 0 should produce color 0, got {color}"
        assert isinstance(edge_indices,
                          list), "Should return list of edge indices"
        assert len(edge_indices) > 0, "Should return non-empty list"

        color, edge_indices = strategy.decode_action(env, action=2)
        assert color == 1, f"Action 2 should produce color 1, got {color}"

    def test_decode_action_color_mapping(self):
        """Test color mapping for multiple actions."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=5, n_edges=10, n_colors=2, n_chord_lengths=2)

        # Action space = 2 * 2 = 4
        colors = []
        for action in range(4):
            color, _ = strategy.decode_action(env, action)
            colors.append(color)

        # Should be [0, 0, 1, 1]
        assert colors == [0, 0, 1, 1], \
            f"Colors for actions 0-3 should be [0,0,1,1], got {colors}"

    def test_decode_action_chord_lengths(self):
        """Test that different actions produce different chord lengths."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=5, n_edges=10, n_colors=2, n_chord_lengths=2)

        # Actions 0 and 1 should produce different edge sets (chord 1 vs chord
        # 2)
        _, edges_0 = strategy.decode_action(env, action=0)
        _, edges_1 = strategy.decode_action(env, action=1)

        # Convert to actual edges
        actual_edges_0 = {env.all_edges[idx] for idx in edges_0}
        actual_edges_1 = {env.all_edges[idx] for idx in edges_1}

        assert actual_edges_0 != actual_edges_1, \
            "Different actions should produce different edge sets"

    def test_missing_n_chord_lengths_attribute(self):
        """Test that missing n_chord_lengths raises AttributeError."""
        strategy = action_types.CirculantActionStrategy()

        # Create env without n_chord_lengths
        class MinimalEnv:
            n_vertices = 4
            n_colors = 2

        env = MinimalEnv()

        with pytest.raises(AttributeError):
            strategy.decode_action(env, action=0)

    def test_action_produces_multiple_edges(self):
        """Test that single action can modify multiple edges simultaneously."""
        strategy = action_types.CirculantActionStrategy()
        env = MockEnv(n_vertices=4, n_edges=6, n_colors=2, n_chord_lengths=2)

        _, edge_indices = strategy.decode_action(env, action=0)

        # Chord length patterns typically modify multiple edges
        assert isinstance(edge_indices, list), "Should return list"
        # For n=4, chord 1 has 4 edges
        assert len(edge_indices) >= 3, \
            "Circulant pattern should affect multiple edges"


class TestActionStrategyInterfaces:
    """Test that all strategies implement the BaseActionStrategy interface."""

    def test_default_strategy_inherits_base(self):
        """Test DefaultActionStrategy inherits from BaseActionStrategy."""
        strategy = action_types.DefaultActionStrategy()
        assert isinstance(strategy, action_types.BaseActionStrategy)

    def test_two_action_strategy_inherits_base(self):
        """Test TwoActionStrategy inherits from BaseActionStrategy."""
        strategy = action_types.TwoActionStrategy()
        assert isinstance(strategy, action_types.BaseActionStrategy)

    def test_circulant_strategy_inherits_base(self):
        """Test CirculantActionStrategy inherits from BaseActionStrategy."""
        strategy = action_types.CirculantActionStrategy()
        assert isinstance(strategy, action_types.BaseActionStrategy)

    def test_all_strategies_have_decode_action(self):
        """Test all strategies implement decode_action method."""
        strategies = [
            action_types.DefaultActionStrategy(),
            action_types.TwoActionStrategy(),
            action_types.CirculantActionStrategy(),
        ]

        for strategy in strategies:
            assert hasattr(strategy, "decode_action"), \
                f"{strategy.__class__.__name__} should have decode_action \
                    method"
            assert callable(strategy.decode_action), \
                f"{strategy.__class__.__name__}.decode_action should be \
                    callable"
