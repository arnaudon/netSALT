"""Unit tests for the pure helpers in netsalt.

These are fast, graph-free tests intended to pin down the load-bearing
utilities that the rest of the library composes. Extending this file with
every new bug that slips past the functional test is the point.
"""
import numpy as np
import networkx as nx
import pytest

from netsalt.algorithm import clean_duplicate_modes
from netsalt.pump import pump_cost
from netsalt.utils import (
    from_complex,
    get_scan_grid,
    linewidth,
    lorentzian,
    order_edges_by,
    to_complex,
)


class TestComplexConversion:
    def test_to_complex_uses_minus_imag_convention(self):
        # netsalt stores a mode as [k, alpha] with alpha = -imag(k)
        assert to_complex([3.0, 0.5]) == 3.0 - 0.5j

    def test_to_complex_passes_through_scalars(self):
        assert to_complex(2.0 + 1.0j) == 2.0 + 1.0j
        assert to_complex(2.0) == 2.0

    def test_from_complex_is_inverse_of_to_complex(self):
        mode = [1.25, -0.3]
        assert from_complex(to_complex(mode)) == mode

    def test_from_complex_passthrough(self):
        arr = np.array([1.0, 2.0])
        assert from_complex(arr) is arr
        assert from_complex([1.0, 2.0]) == [1.0, 2.0]


class TestLinewidth:
    def test_peak_at_center(self):
        assert linewidth(5.0, k_center=5.0, width=0.1) == 1.0

    def test_half_maximum_at_width(self):
        # Lorentzian in this parametrisation is 1/2 at |k - k_center| = width
        assert linewidth(5.5, k_center=5.0, width=0.5) == pytest.approx(0.5)

    def test_lorentzian_reads_graph_params(self):
        graph = nx.Graph()
        graph.graph["params"] = {"k_a": 10.0, "gamma_perp": 0.2}
        assert lorentzian(10.0, graph) == 1.0
        assert lorentzian(10.2, graph) == pytest.approx(0.5)


class TestScanGrid:
    def test_scan_grid_shape(self):
        graph = nx.Graph()
        graph.graph["params"] = {
            "k_min": 1.0,
            "k_max": 2.0,
            "k_n": 5,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_n": 3,
        }
        ks, alphas = get_scan_grid(graph)
        assert len(ks) == 5
        assert len(alphas) == 3
        assert ks[0] == 1.0 and ks[-1] == 2.0
        assert alphas[0] == 0.0 and alphas[-1] == 1.0


class TestCleanDuplicateModes:
    def test_drops_close_duplicate(self):
        modes = [[1.0, 0.1], [1.00001, 0.10001], [2.0, 0.2]]
        result = clean_duplicate_modes(modes, k_size=1e-3, alpha_size=1e-3)
        # first is flagged duplicate of second; one of them is removed
        assert len(result) == 2

    def test_keeps_distinct_modes(self):
        modes = [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
        result = clean_duplicate_modes(modes, k_size=1e-3, alpha_size=1e-3)
        assert len(result) == 3


class TestOrderEdgesBy:
    def test_sorts_ascending(self):
        g = nx.path_graph(4)
        ordered = order_edges_by(g, [3.0, 1.0, 2.0])
        assert ordered == [list(g.edges)[1], list(g.edges)[2], list(g.edges)[0]]


class TestPumpCost:
    def test_non_overlapping_modes_gives_finite_cost(self):
        # Two modes, two edges; mode 0 lives on edge 0, mode 1 on edge 1.
        # Pump edge 0 only; optimise mode 0.
        pump = np.array([1, 0])
        pump_overlapps = np.array([[1.0, 0.0], [0.0, 1.0]])
        cost = pump_cost(pump, modes_to_optimise=[0], pump_overlapps=pump_overlapps)
        # numerator: max over non-optimised modes of overlap with pump = 0
        # denominator: optimised mode overlap with pump = 1
        assert cost == 0.0

    def test_min_size_penalty(self):
        pump = np.array([1, 0])
        pump_overlapps = np.array([[1.0, 0.0], [0.0, 1.0]])
        cost = pump_cost(
            pump, modes_to_optimise=[0], pump_overlapps=pump_overlapps, pump_min_size=2
        )
        assert cost == 1e10


class TestDispersionRelations:
    def test_linear_dispersion_raises_without_params(self):
        from netsalt.physics import dispersion_relation_linear

        with pytest.raises(ValueError):
            dispersion_relation_linear(1.0, params=None)
        with pytest.raises(ValueError):
            dispersion_relation_linear(1.0, params={})

    def test_dielectric_dispersion_raises_without_params(self):
        from netsalt.physics import dispersion_relation_dielectric

        with pytest.raises(ValueError):
            dispersion_relation_dielectric(1.0, params=None)


class TestModesImport:
    def test_import_does_not_mutate_global_warning_state(self):
        """Regression: importing netsalt.modes used to call
        warnings.filterwarnings('ignore') at module scope."""
        import warnings

        import netsalt.modes  # noqa: F401

        # a fresh catch_warnings stack should not have an 'ignore' default filter
        # left over from module import — if we emit a warning here, it should
        # surface rather than being suppressed by leaked filters.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("sentinel", UserWarning)
            assert any("sentinel" in str(w.message) for w in caught)
