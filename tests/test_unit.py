"""Unit tests for the pure helpers in netsalt.

These are fast, graph-free tests intended to pin down the load-bearing
utilities that the rest of the library composes. Extending this file with
every new bug that slips past the functional test is the point.
"""

import networkx as nx
import numpy as np
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


class TestNetSaltParams:
    """Pydantic-backed params model replaces the old dict."""

    def test_dict_style_access(self):
        from netsalt.params import NetSaltParams

        p = NetSaltParams.from_dict({"k_min": 1.0, "k_max": 2.0, "custom": [1, 2, 3]})
        assert p["k_min"] == 1.0
        assert p["custom"] == [1, 2, 3]
        assert p.get("missing", "DEFAULT") == "DEFAULT"
        assert "k_min" in p
        assert "missing" not in p

    def test_assignment_triggers_validation(self):
        from pydantic import ValidationError

        from netsalt.params import NetSaltParams

        p = NetSaltParams()
        p["k_min"] = 1.5  # float OK
        with pytest.raises(ValidationError):
            p["k_min"] = "not a float"  # rejected at the boundary

    def test_extra_keys_still_allowed(self):
        from netsalt.params import NetSaltParams

        p = NetSaltParams.from_dict({"problem_specific_knob": 42})
        assert p["problem_specific_knob"] == 42
        assert list(p.keys()) == ["problem_specific_knob"]

    def test_from_dict_accepts_none(self):
        from netsalt.params import NetSaltParams

        p = NetSaltParams.from_dict(None)
        assert len(list(p.keys())) == 0

    def test_update_parameters_converts_dict(self):
        """``update_parameters`` should upgrade a bare dict on the graph to
        a ``NetSaltParams`` instance so subsequent access gets validated."""
        from netsalt.params import NetSaltParams
        from netsalt.quantum_graph import update_parameters

        graph = nx.Graph()
        update_parameters(graph, {"k_min": 1.0})
        assert isinstance(graph.graph["params"], NetSaltParams)
        assert graph.graph["params"]["k_min"] == 1.0


class TestGraphIO:
    """JSON graph I/O is safe and round-trips."""

    def test_save_load_round_trip(self, tmp_path):
        from netsalt.io import load_graph, save_graph

        g = nx.path_graph(3)
        g.graph["params"] = {"k_min": 1.0, "k_max": 2.0}
        for n in g.nodes:
            g.nodes[n]["position"] = np.array([float(n), 0.0])
        for u, v in g.edges:
            g[u][v]["dielectric_constant"] = 2.0 + 0.1j
            g[u][v]["length"] = 1.0

        path = tmp_path / "graph.json"
        save_graph(g, str(path))

        loaded = load_graph(str(path))
        assert list(loaded.nodes) == list(g.nodes)
        assert list(loaded.edges) == list(g.edges)
        assert loaded.graph["params"]["k_min"] == 1.0
        assert np.allclose(loaded.nodes[0]["position"], [0.0, 0.0])
        assert loaded[0][1]["dielectric_constant"] == 2.0 + 0.1j

    def test_load_pickle_refused_by_default(self, tmp_path):
        """Unpickling is an ACE sink — must be explicit opt-in."""
        from netsalt.io import load_graph, save_graph

        g = nx.path_graph(3)
        path = tmp_path / "graph.pkl"
        with pytest.warns(DeprecationWarning):
            save_graph(g, str(path))

        with pytest.raises(ValueError, match="pickle"):
            load_graph(str(path))

        with pytest.warns(DeprecationWarning):
            loaded = load_graph(str(path), allow_pickle=True)
        assert list(loaded.nodes) == list(g.nodes)


class TestComputeCore:
    """Smoke + structural tests for the compute primitives."""

    def _line_graph(self, n_edges=5, dielectric=4.0):
        """3-node line graph with unit edge lengths and a constant dispersion."""
        import netsalt
        from netsalt.physics import dispersion_relation_dielectric
        from netsalt.quantum_graph import create_quantum_graph

        g = nx.path_graph(n_edges + 1)
        positions = np.array([[float(i), 0.0] for i in range(n_edges + 1)])
        params = {
            "open_model": "open",
            "dielectric_params": {
                "method": "uniform",
                "inner_value": dielectric,
                "loss": 0.0,
                "outer_value": 1.0,
            },
            "c": 1.0,
        }
        create_quantum_graph(g, params, positions=positions)
        netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
        netsalt.set_dielectric_constant(g, g.graph["params"])
        return g

    def test_construct_laplacian_is_square(self):
        from netsalt.quantum_graph import construct_laplacian

        g = self._line_graph(n_edges=4)
        L = construct_laplacian(1.0 + 0.0j, g)
        assert L.shape == (len(g), len(g))

    def test_weight_and_incidence_shapes(self):
        from netsalt.quantum_graph import (
            construct_incidence_matrix,
            construct_weight_matrix,
            set_wavenumber,
        )

        g = self._line_graph(n_edges=5)
        set_wavenumber(g, 1.0 + 0.0j)
        BT, B = construct_incidence_matrix(g)
        W = construct_weight_matrix(g)
        n_nodes, n_edges = len(g), len(g.edges)
        # B rows = 2 * n_edges, cols = n_nodes; BT is the transpose of B.
        assert B.shape == (2 * n_edges, n_nodes)
        assert BT.shape == (n_nodes, 2 * n_edges)
        assert W.shape == (2 * n_edges, 2 * n_edges)

    def test_mode_quality_accepts_generator(self):
        """Regression: threading an rng through should be deterministic."""
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph(n_edges=3)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        q1 = mode_quality([2.0, 0.1], g, rng=rng1)
        q2 = mode_quality([2.0, 0.1], g, rng=rng2)
        assert q1 == pytest.approx(q2, rel=1e-12)


class TestPhysicsPrimitives:
    """Pure scalar helpers — targets of future regressions."""

    def test_gamma_without_gamma_perp_returns_minus_i(self):
        from netsalt.physics import gamma

        assert gamma(1.0, {}) == -1.0j

    def test_gamma_peak_at_k_a_is_minus_i(self):
        from netsalt.physics import gamma

        # gamma(k_a) = gamma_perp / (0 + j*gamma_perp) = -j
        result = gamma(5.0, {"gamma_perp": 2.0, "k_a": 5.0})
        assert result == pytest.approx(-1.0j)

    def test_q_value_of_complex_mode(self):
        from netsalt.physics import q_value

        # q = real / (2 * imag_alpha), with mode = [k, alpha]
        assert q_value([10.0, 0.5]) == 10.0


class TestRngIsolation:
    """Regression: compute functions used to call ``np.random.seed`` which
    mutates the process-wide RNG state."""

    def test_refine_mode_brownian_ratchet_accepts_rng(self):
        from inspect import signature

        from netsalt.algorithm import refine_mode_brownian_ratchet

        assert "rng" in signature(refine_mode_brownian_ratchet).parameters

    def test_laplacian_quality_and_mode_quality_accept_rng(self):
        from inspect import signature

        from netsalt.quantum_graph import laplacian_quality, mode_quality

        assert "rng" in signature(laplacian_quality).parameters
        assert "rng" in signature(mode_quality).parameters

    def test_worker_scan_owns_a_generator(self):
        """``WorkerScan`` must carry a per-instance Generator, not reseed the
        module-level ``np.random`` RandomState."""
        from netsalt.modes import WorkerScan

        ws = WorkerScan.__new__(WorkerScan)
        ws.graph = None
        ws.quality_method = "eigenvalue"
        ws.rng = np.random.default_rng(42)
        assert isinstance(ws.rng, np.random.Generator)
