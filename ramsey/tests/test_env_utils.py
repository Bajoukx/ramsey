"""Tests for ramsey.env_utils module.

This module tests the foundational utility functions for adjacency matrix
manipulation, initialization methods, and conversions between different
graph representations.
"""

import pytest
import torch
import itertools

from ramsey import env_utils


class TestFlattenAdjacencyMatrix:
    """Tests for flaten_adjacency_matrix function."""

    def test_flatten_shape_for_n3(self):
        """Test flatten produces correct length for 3x3 matrix."""
        n = 3
        matrix = torch.ones(n, n)
        result = env_utils.flaten_adjacency_matrix(matrix)

        expected_length = n * (n - 1) // 2
        assert result.shape == (expected_length,), \
            f"Expected length {expected_length}, got {result.shape[0]}"
        assert result.numel() == 3, "3x3 matrix should produce 3 elements"

    def test_flatten_shape_for_n4(self):
        """Test flatten produces correct length for 4x4 matrix."""
        n = 4
        matrix = torch.ones(n, n)
        result = env_utils.flaten_adjacency_matrix(matrix)

        expected_length = n * (n - 1) // 2
        assert result.shape == (expected_length,), \
            f"Expected length {expected_length}, got {result.shape[0]}"
        assert result.numel() == 6, "4x4 matrix should produce 6 elements"

    def test_flatten_ordering_matches_combinations(self):
        """Test flatten ordering matches
        itertools.combinations((0,1),(0,2),(1,2))."""
        # Create matrix with distinct upper-triangular values
        matrix = torch.tensor([
            [0, 1, 2],
            [0, 0, 3],
            [0, 0, 0],
        ],
                              dtype=torch.float32)

        result = env_utils.flaten_adjacency_matrix(matrix)

        # Expected order: (0,1)=1, (0,2)=2, (1,2)=3
        expected = torch.tensor([1, 2, 3], dtype=torch.int64)
        assert torch.equal(result, expected), \
            f"Expected {expected}, got {result}"

    def test_flatten_dtype_is_int64(self):
        """Test flatten returns int64 dtype (via .long())."""
        matrix = torch.ones(3, 3, dtype=torch.float32)
        result = env_utils.flaten_adjacency_matrix(matrix)

        assert result.dtype == torch.int64, \
            f"Expected dtype int64, got {result.dtype}"

    def test_flatten_ignores_diagonal(self):
        """Test flatten excludes diagonal elements."""
        matrix = torch.tensor([
            [99, 1, 2],
            [1, 99, 3],
            [2, 3, 99],
        ],
                              dtype=torch.float32)

        result = env_utils.flaten_adjacency_matrix(matrix)

        # Diagonal (99) should not appear in result
        assert 99 not in result, "Diagonal elements should be excluded"
        expected = torch.tensor([1, 2, 3], dtype=torch.int64)
        assert torch.equal(result, expected), \
            f"Expected {expected}, got {result}"


class TestUnflattenVecToAdjacencyMatrix:
    """Tests for unflaten_vec_to_adjacency_matrix function."""

    def test_unflatten_shape_from_length_3(self):
        """Test unflatten infers n=3 from length 3."""
        vec = torch.tensor([1, 2, 3])
        result = env_utils.unflaten_vec_to_adjacency_matrix(vec)

        assert result.shape == (3, 3), \
            f"Length 3 should produce 3x3 matrix, got {result.shape}"

    def test_unflatten_shape_from_length_6(self):
        """Test unflatten infers n=4 from length 6."""
        vec = torch.tensor([1, 2, 3, 4, 5, 6])
        result = env_utils.unflaten_vec_to_adjacency_matrix(vec)

        assert result.shape == (4, 4), \
            f"Length 6 should produce 4x4 matrix, got {result.shape}"

    def test_unflatten_is_symmetric(self):
        """Test unflatten produces symmetric matrix."""
        vec = torch.tensor([1, 2, 3])
        result = env_utils.unflaten_vec_to_adjacency_matrix(vec)

        assert torch.equal(result, result.T), \
            "Result should be symmetric"

    def test_unflatten_diagonal_is_zero(self):
        """Test unflatten sets diagonal to zero."""
        vec = torch.tensor([1, 2, 3, 4, 5, 6])
        result = env_utils.unflaten_vec_to_adjacency_matrix(vec)

        diagonal = torch.diag(result)
        assert torch.all(diagonal == 0), \
            f"Diagonal should be all zeros, got {diagonal}"

    def test_unflatten_preserves_upper_triangle_values(self):
        """Test unflatten correctly places values in upper triangle."""
        vec = torch.tensor([1, 2, 3])
        result = env_utils.unflaten_vec_to_adjacency_matrix(vec)

        # Upper triangle values: (0,1)=1, (0,2)=2, (1,2)=3
        assert result[0, 1] == 1, "Position (0,1) should be 1"
        assert result[0, 2] == 2, "Position (0,2) should be 2"
        assert result[1, 2] == 3, "Position (1,2) should be 3"

        # Check symmetry
        assert result[1, 0] == 1, "Position (1,0) should mirror (0,1)"
        assert result[2, 0] == 2, "Position (2,0) should mirror (0,2)"
        assert result[2, 1] == 3, "Position (2,1) should mirror (1,2)"


class TestFlattenUnflattenRoundtrip:
    """Tests for flatten/unflatten roundtrip correctness."""

    def test_roundtrip_preserves_structure_n3(self):
        """Test flatten -> unflatten roundtrip for 3x3 matrix."""
        original = torch.tensor([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ],
                                dtype=torch.float32)

        flattened = env_utils.flaten_adjacency_matrix(original)
        restored = env_utils.unflaten_vec_to_adjacency_matrix(flattened)

        # Compare upper triangles (dtype may differ)
        assert torch.equal(restored[0, 1], original[0, 1]), \
            "Position (0,1) should be preserved"
        assert torch.equal(restored[0, 2], original[0, 2]), \
            "Position (0,2) should be preserved"
        assert torch.equal(restored[1, 2], original[1, 2]), \
            "Position (1,2) should be preserved"

        # Check symmetry
        assert torch.equal(restored, restored.T), \
            "Restored matrix should be symmetric"

    def test_roundtrip_preserves_structure_n5(self):
        """Test roundtrip for larger 5x5 matrix."""
        n = 5
        # Create upper triangular values
        original = torch.zeros(n, n)
        edges = list(itertools.combinations(range(n), 2))
        for idx, (i, j) in enumerate(edges):
            original[i, j] = idx
            original[j, i] = idx  # Symmetric

        flattened = env_utils.flaten_adjacency_matrix(original)
        restored = env_utils.unflaten_vec_to_adjacency_matrix(flattened)

        # Check all upper triangle values
        for i, j in edges:
            assert restored[i, j] == original[i, j], \
                f"Position ({i},{j}) should be preserved"


class TestAdjVecToDict:
    """Tests for adj_vec_to_dict function."""

    def test_single_color_graph(self):
        """Test conversion of single-color graph."""
        # Triangle with all edges color 0
        adjacency_vec = torch.tensor([0, 0, 0])
        result = env_utils.adj_vec_to_dict(adjacency_vec, color=0)

        # Should include all 3 vertices
        assert len(result) == 3, f"Expected 3 vertices, got {len(result)}"
        assert "0" in result, "Vertex 0 should be present"
        assert "1" in result, "Vertex 1 should be present"
        assert "2" in result, "Vertex 2 should be present"

        # Check adjacencies
        assert "1" in result["0"] and "2" in result["0"], \
            "Vertex 0 should be connected to 1 and 2"
        assert "0" in result["1"] and "2" in result["1"], \
            "Vertex 1 should be connected to 0 and 2"
        assert "0" in result["2"] and "1" in result["2"], \
            "Vertex 2 should be connected to 0 and 1"

    def test_color_filtering(self):
        """Test that only edges with specified color are included."""
        # Edges: (0,1)=0, (0,2)=1, (1,2)=-1
        adjacency_vec = torch.tensor([0, 1, -1])

        # Filter for color 0
        result_0 = env_utils.adj_vec_to_dict(adjacency_vec, color=0)
        assert "0" in result_0 and "1" in result_0, \
            "Color 0: should include vertices 0,1"
        assert "2" not in result_0, \
            "Color 0: vertex 2 has no incident edges of this color"

        # Filter for color 1
        result_1 = env_utils.adj_vec_to_dict(adjacency_vec, color=1)
        assert "0" in result_1 and "2" in result_1, \
            "Color 1: should include vertices 0,2"
        assert "1" not in result_1, \
            "Color 1: vertex 1 has no incident edge (0,2)"

        # Filter for color -1 (uncolored)
        result_neg1 = env_utils.adj_vec_to_dict(adjacency_vec, color=-1)
        assert "1" in result_neg1 and "2" in result_neg1, \
            "Color -1: should include vertices 1,2"
        assert "0" not in result_neg1, \
            "Color -1: vertex 0 has no incident uncolored edges"

    def test_isolated_vertices_omitted(self):
        """Test that vertices with no incident edges of color are omitted."""
        # 4 vertices, only edge (0,1) has color 0
        # Edges: (0,1)=0, (0,2)=1, (0,3)=1, (1,2)=1, (1,3)=1, (2,3)=1
        adjacency_vec = torch.tensor([0, 1, 1, 1, 1, 1])

        result = env_utils.adj_vec_to_dict(adjacency_vec, color=0)

        assert "0" in result and "1" in result, \
            "Vertices 0,1 should be present (edge 0-1 is color 0)"
        assert "2" not in result and "3" not in result, \
            "Vertices 2,3 should be omitted (no incident color-0 edges)"

    def test_empty_graph(self):
        """Test empty adjacency vector returns empty dict."""
        adjacency_vec = torch.tensor([])
        result = env_utils.adj_vec_to_dict(adjacency_vec, color=0)

        assert len(result) == 0, "Empty vector should produce empty dict"


class TestAdjMatrixToDict:
    """Tests for adj_matrix_to_dict function."""

    def test_matrix_to_dict_single_color(self):
        """Test conversion from adjacency matrix to dict."""
        matrix = torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
                              dtype=torch.float32)

        result = env_utils.adj_matrix_to_dict(matrix, color=1)

        # All edges are color 1 (off-diagonal)
        assert len(result) == 3, f"Expected 3 vertices, got {len(result)}"
        assert "0" in result and "1" in result and "2" in result, \
            "All vertices should be present"

    def test_matrix_to_dict_sparse(self):
        """Test sparse matrix (some vertices isolated for given color)."""
        matrix = torch.tensor([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
                              dtype=torch.float32)

        result = env_utils.adj_matrix_to_dict(matrix, color=1)

        # Only edges (0,1) and (2,3) are color 1
        assert "0" in result and "1" in result, \
            "Vertices 0,1 should be present"
        assert "2" in result and "3" in result, \
            "Vertices 2,3 should be present"


class TestInitMethods:
    """Tests for initialization functions."""

    def test_init_empty(self):
        """Test init_empty returns all zeros."""

        class MockEnv:
            n_edges = 6

        env = MockEnv()
        result = env_utils.init_empty(env)

        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"
        assert torch.all(result == 0), "All values should be 0"
        assert result.dtype == torch.float32, \
            f"Expected float32, got {result.dtype}"

    def test_init_uncolored(self):
        """Test init_uncolored returns all -1."""

        class MockEnv:
            n_edges = 6

        env = MockEnv()
        result = env_utils.init_uncolored(env)

        assert result.shape == (6,), f"Expected shape (6,), got {result.shape}"
        assert torch.all(result == -1), "All values should be -1"
        assert result.dtype == torch.float32, \
            f"Expected float32, got {result.dtype}"

    def test_get_all_init_methods(self):
        """Test get_all_init_methods returns dict with known methods."""
        methods = env_utils.get_all_init_methods()

        assert isinstance(methods, dict), "Should return a dict"
        assert "empty" in methods, "Should include 'empty' method"
        assert "uncolored" in methods, "Should include 'uncolored' method"
        assert callable(methods["empty"]), "'empty' should be callable"
        assert callable(methods["uncolored"]), "'uncolored' should be callable"

    def test_get_init_function_known_method(self):
        """Test get_init_function returns correct function."""
        func_empty = env_utils.get_init_function("empty")
        func_uncolored = env_utils.get_init_function("uncolored")

        assert callable(func_empty), "Should return callable for 'empty'"
        assert callable(
            func_uncolored), "Should return callable for 'uncolored'"
        assert func_empty is env_utils.init_empty, \
            "Should return the actual init_empty function"
        assert func_uncolored is env_utils.init_uncolored, \
            "Should return the actual init_uncolored function"

    def test_get_init_function_unknown_method_raises(self):
        """Test get_init_function raises KeyError for unknown method."""
        with pytest.raises(KeyError):
            env_utils.get_init_function("nonexistent_method")
