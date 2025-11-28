"""Test for clique counting algorithms"""

from ramsey import clique_algorithms


class TestCliqueCounters:
    """Test clique counting algorithms."""

    _TEST_GRAPH = {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2, 4},
        4: {3, 5, 6},
        5: {4, 6},
        6: {4, 5},
    }

    def test_bron_kerbosch_number_of_cliques(self):
        """Test Bron-Kerbosch algorithm.
        
        This test verifies that the Bron–Kerbosch algorithm identifies
        the number of cliques >= 3 in the graph. The example graph contains
        two cliques: {0, 1, 2, 3} and {4, 5, 6}.
        """
        cliques = clique_algorithms.bron_kerbosch(self._TEST_GRAPH)
        assert len(cliques) == 2

    def test_bron_kerbosch_clique_max_cliques(self):
        """Test Bron-Kerbosch algorithm.

        This test verifies that the Bron–Kerbosch algorithm identifies
        the correct maximal cliques in the graph. The example graph contains
        two maximalcliques: {0, 1, 2, 3} and {4, 5, 6}.
        """
        cliques = clique_algorithms.bron_kerbosch(self._TEST_GRAPH)
        expected = {frozenset({0, 1, 2, 3}), frozenset({4, 5, 6})}
        assert set(map(frozenset, cliques)) == expected
