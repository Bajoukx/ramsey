"""Tests for ramsey.clique_algorithms module.

This module tests the Bron-Kerbosch algorithm for maximal clique enumeration.
The implementation filters cliques to only return those with size > 2 (triangles
and larger).
"""

from ramsey import clique_algorithms


class TestBronKerbosch:
    """Tests for bron_kerbosch maximal clique finding algorithm."""

    def test_single_triangle(self):
        """Test detection of a single triangle (K3)."""
        # Triangle: vertices 0-1-2 all connected
        graph_dict = {
            "0": {"1", "2"},
            "1": {"0", "2"},
            "2": {"0", "1"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should find exactly one maximal clique of size 3
        assert len(result) == 1, f"Expected 1 clique, found {len(result)}"

        # Convert to set for order-insensitive comparison
        clique_set = set(result[0])
        expected = {"0", "1", "2"}
        assert clique_set == expected, \
            f"Expected clique {expected}, got {clique_set}"

    def test_k4_complete_graph(self):
        """Test detection of K4 (complete graph on 4 vertices)."""
        # All vertices connected to all others
        graph_dict = {
            "0": {"1", "2", "3"},
            "1": {"0", "2", "3"},
            "2": {"0", "1", "3"},
            "3": {"0", "1", "2"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # K4 has one maximal clique of size 4
        assert len(result) == 1, f"Expected 1 clique, found {len(result)}"

        clique_set = set(result[0])
        expected = {"0", "1", "2", "3"}
        assert clique_set == expected, \
            f"Expected clique {expected}, got {clique_set}"
        assert len(clique_set) == 4, "K4 should have one 4-clique"

    def test_two_overlapping_triangles(self):
        """Test detection of two triangles sharing an edge."""
        # Triangles: (0,1,2) and (1,2,3)
        graph_dict = {
            "0": {"1", "2"},
            "1": {"0", "2", "3"},
            "2": {"0", "1", "3"},
            "3": {"1", "2"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should find two maximal triangles
        assert len(result) == 2, f"Expected 2 cliques, found {len(result)}"

        # Convert to sets for comparison (order-insensitive)
        cliques_as_sets = [set(clique) for clique in result]
        expected_cliques = [{"0", "1", "2"}, {"1", "2", "3"}]

        for expected in expected_cliques:
            assert expected in cliques_as_sets, \
                f"Expected clique {expected} not found in {cliques_as_sets}"

    def test_two_disjoint_triangles(self):
        """Test detection of two separate triangles with no shared vertices."""
        # Triangle 1: (0,1,2), Triangle 2: (3,4,5)
        graph_dict = {
            "0": {"1", "2"},
            "1": {"0", "2"},
            "2": {"0", "1"},
            "3": {"4", "5"},
            "4": {"3", "5"},
            "5": {"3", "4"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should find two disjoint triangles
        assert len(result) == 2, f"Expected 2 cliques, found {len(result)}"

        cliques_as_sets = [set(clique) for clique in result]
        expected_cliques = [{"0", "1", "2"}, {"3", "4", "5"}]

        for expected in expected_cliques:
            assert expected in cliques_as_sets, \
                f"Expected clique {expected} not found in {cliques_as_sets}"

    def test_no_triangles_returns_empty(self):
        """Test that graphs with no triangles return empty list.
        
        The implementation filters out cliques with size <= 2.
        A 4-cycle has maximal cliques of size 2 only.
        """
        # 4-cycle: 0-1-2-3-0 (no triangles)
        graph_dict = {
            "0": {"1", "3"},
            "1": {"0", "2"},
            "2": {"1", "3"},
            "3": {"2", "0"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should return empty list (no cliques > size 2)
        assert not result, \
            f"Expected empty list for graph with no triangles, got {result}"

    def test_path_graph_returns_empty(self):
        """Test that a path graph (no triangles) returns empty list."""
        # Path: 0-1-2-3
        graph_dict = {
            "0": {"1"},
            "1": {"0", "2"},
            "2": {"1", "3"},
            "3": {"2"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        assert not result, f"Expected empty list for path graph, got {result}"

    def test_empty_graph_returns_empty(self):
        """Test that an empty graph returns empty list."""
        graph_dict = {}

        result = clique_algorithms.bron_kerbosch(graph_dict)

        assert not result, "Expected empty list for empty graph"

    def test_single_vertex_returns_empty(self):
        """Test that a single isolated vertex returns empty list."""
        # Single vertex with no edges
        graph_dict = {"0": set()}

        result = clique_algorithms.bron_kerbosch(graph_dict)

        assert not result, \
            "Expected empty list for single vertex (clique size 1 filtered)"

    def test_single_edge_returns_empty(self):
        """Test that a single edge returns empty list."""
        # Single edge: 0-1
        graph_dict = {
            "0": {"1"},
            "1": {"0"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        assert not result, \
            "Expected empty list for single edge (clique size 2 filtered)"

    def test_k5_complete_graph(self):
        """Test detection of K5 (complete graph on 5 vertices)."""
        # All 5 vertices connected to all others
        graph_dict = {
            "0": {"1", "2", "3", "4"},
            "1": {"0", "2", "3", "4"},
            "2": {"0", "1", "3", "4"},
            "3": {"0", "1", "2", "4"},
            "4": {"0", "1", "2", "3"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # K5 has one maximal clique of size 5
        assert len(result) == 1, f"Expected 1 clique, found {len(result)}"

        clique_set = set(result[0])
        expected = {"0", "1", "2", "3", "4"}
        assert clique_set == expected, \
            f"Expected clique {expected}, got {clique_set}"
        assert len(clique_set) == 5, "K5 should have one 5-clique"

    def test_diamond_graph(self):
        """Test diamond graph (K4 minus one edge).
        
        Has two overlapping triangles sharing an edge.
        """
        # Diamond: 0-1-2-3 with 0-2 and 1-3 edges (missing 0-3)
        #   0
        #  /|\
        # 1 | 2
        #  \|/
        #   3
        # But missing edge 0-3
        graph_dict = {
            "0": {"1", "2"},
            "1": {"0", "2", "3"},
            "2": {"0", "1", "3"},
            "3": {"1", "2"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should find two maximal triangles: (0,1,2) and (1,2,3)
        assert len(result) == 2, f"Expected 2 cliques, found {len(result)}"

        cliques_as_sets = [set(clique) for clique in result]
        # Both triangles should be present
        for clique in cliques_as_sets:
            assert len(
                clique) == 3, f"Expected triangles (size 3), got {clique}"

    def test_order_insensitive(self):
        """Test that result comparison should be order-insensitive.
        
        This test documents that the order of vertices within cliques
        and the order of cliques themselves may vary.
        """
        graph_dict = {
            "0": {"1", "2"},
            "1": {"0", "2"},
            "2": {"0", "1"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Convert result to set of frozensets for order-insensitive comparison
        result_as_set_of_sets = {frozenset(clique) for clique in result}
        expected_as_set_of_sets = {frozenset({"0", "1", "2"})}

        assert result_as_set_of_sets == expected_as_set_of_sets, \
            "Clique comparison should be order-insensitive"

    def test_house_graph(self):
        """Test house-shaped graph (pentagon with one chord).
        
        Has one triangle and additional structure.
        """
        # Pentagon 0-1-2-3-4-0 with chord 0-2
        # Forms triangle (0,1,2) plus additional edges
        graph_dict = {
            "0": {"1", "2", "4"},
            "1": {"0", "2"},
            "2": {"0", "1", "3"},
            "3": {"2", "4"},
            "4": {"3", "0"},
        }

        result = clique_algorithms.bron_kerbosch(graph_dict)

        # Should find at least the triangle (0,1,2)
        assert len(result) >= 1, "Expected at least one clique"

        # Verify triangle is present
        cliques_as_sets = [set(clique) for clique in result]
        triangle = {"0", "1", "2"}
        assert triangle in cliques_as_sets, \
            f"Expected triangle {triangle} in {cliques_as_sets}"
