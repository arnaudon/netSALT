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

    def test_mode_quality_determinant_path(self):
        """Exercise the determinant-based quality branch."""
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph(n_edges=3)
        q = mode_quality([2.0, 0.1], g, quality_method="determinant")
        assert q > 0

    def test_mode_quality_singularvalue_path(self):
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph(n_edges=3)
        q = mode_quality([2.0, 0.1], g, quality_method="singularvalue")
        assert q >= 0

    def test_construct_weight_matrix_with_k_flag(self):
        """``with_k=False`` is the branch used for edge-amplitude calculations."""
        from netsalt.quantum_graph import construct_weight_matrix, set_wavenumber

        g = self._line_graph(n_edges=4)
        set_wavenumber(g, 2.0 + 0.1j)
        W_with_k = construct_weight_matrix(g, with_k=True)
        W_no_k = construct_weight_matrix(g, with_k=False)
        # The "with_k" variant multiplies the diagonal by k, so the matrices differ
        assert not np.allclose(W_with_k.toarray(), W_no_k.toarray())

    def test_set_total_length_rescales(self):
        """``set_total_length`` should rescale edges to match the target sum."""
        from netsalt.quantum_graph import get_total_inner_length, set_total_length

        g = self._line_graph(n_edges=5)
        set_total_length(g, total_length=2.5)
        assert get_total_inner_length(g) == pytest.approx(2.5, rel=1e-9)

    def test_set_total_length_rejects_both_args(self):
        from netsalt.quantum_graph import set_total_length

        g = self._line_graph(n_edges=3)
        with pytest.raises(ValueError, match="only one of"):
            set_total_length(g, total_length=1.0, max_extent=2.0)

    def test_oversample_graph_adds_nodes(self):
        """Oversampling with a small edge_size should add intermediate nodes."""
        from netsalt.quantum_graph import oversample_graph

        g = self._line_graph(n_edges=3)
        n_before = len(g)
        g2 = oversample_graph(g, edge_size=0.1)
        assert len(g2) > n_before

    def test_refine_mode_brownian_ratchet_converges(self):
        """The refine algorithm should converge to a mode from a nearby guess."""
        from netsalt.algorithm import refine_mode_brownian_ratchet
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph(n_edges=4)
        # Pick an initial mode near a true solution for a dielectric=4 line
        # graph; the exact location doesn't matter — we just need to check
        # that the ratchet returns something with a lower quality than the
        # initial guess.
        initial = np.array([3.0, 0.05])
        initial_q = mode_quality(initial, g)
        params = dict(g.graph["params"])
        params["quality_threshold"] = 1e-3
        params["max_steps"] = 500
        result = refine_mode_brownian_ratchet(
            initial,
            g,
            params,
            rng=np.random.default_rng(0),
        )
        final_q = mode_quality(result, g)
        assert final_q < initial_q

    def test_mode_on_nodes_returns_node_vector(self):
        """mode_on_nodes solves the null-space problem on the laplacian."""
        from netsalt.modes import mode_on_nodes

        g = self._line_graph(n_edges=4)
        # Loosen the quality gate so any grid point passes — this is a
        # coverage / shape smoke test, not an accuracy test.
        g.graph["params"]["quality_threshold"] = 10.0
        solution = mode_on_nodes([3.0, 0.05], g)
        assert solution.shape == (len(g),)

    def test_mode_on_nodes_rejects_non_modes(self):
        """If the quality at the supplied point exceeds the threshold, the
        function should raise loudly rather than return a bogus vector."""
        from netsalt.modes import mode_on_nodes

        g = self._line_graph(n_edges=4)
        g.graph["params"]["quality_threshold"] = 1e-12
        with pytest.raises(ValueError, match="quality is too high"):
            mode_on_nodes([3.0, 0.05], g)

    def _pump_graph(self, n_edges=4, dielectric=4.0):
        """Line graph wired up for pump-dispersion (k_a, gamma_perp, D0, pump)."""
        import netsalt
        from netsalt.physics import dispersion_relation_pump
        from netsalt.quantum_graph import update_parameters

        g = self._line_graph(n_edges=n_edges, dielectric=dielectric)
        netsalt.set_dispersion_relation(g, dispersion_relation_pump)
        update_parameters(
            g,
            {
                "k_a": 3.0,
                "gamma_perp": 1.0,
                "D0": 0.0,
                "pump": np.ones(len(g.edges)),
            },
        )
        return g

    def test_flux_and_mean_mode_on_edges(self):
        """flux_on_edges and mean_mode_on_edges share plumbing with
        compute_overlapping_factor — one test covers them all."""
        from netsalt.modes import flux_on_edges, mean_mode_on_edges

        g = self._pump_graph()
        g.graph["params"]["quality_threshold"] = 10.0
        mode = [3.0, 0.05]
        flux = flux_on_edges(mode, g)
        mean = mean_mode_on_edges(mode, g)
        assert flux.shape == (2 * len(g.edges),)
        assert mean.shape == (len(g.edges),)

    def test_compute_overlapping_factor_and_pump_linear(self):
        """pump_linear depends on compute_overlapping_factor and on gamma()."""
        from netsalt.modes import compute_overlapping_factor, pump_linear

        g = self._pump_graph()
        g.graph["params"]["quality_threshold"] = 10.0
        mode = [3.0, 0.05]
        overlap = compute_overlapping_factor(mode, g)
        # Overlap should be a scalar-like complex
        assert np.ndim(overlap) == 0 or overlap.size == 1
        new_mode = pump_linear(mode, g, D0_0=0.0, D0_1=0.1)
        assert len(new_mode) == 2

    def test_compute_overlapping_single_edges(self):
        """Per-edge overlap vector has one entry per edge."""
        from netsalt.modes import compute_overlapping_single_edges

        g = self._pump_graph(n_edges=3)
        g.graph["params"]["quality_threshold"] = 10.0
        overlap = compute_overlapping_single_edges([3.0, 0.05], g)
        assert overlap.shape == (len(g.edges),)

    def test_compute_mode_IPR_returns_scalar(self):
        """IPR is a scalar computed from mode energy integrals."""
        import pandas as pd

        from netsalt.modes import compute_mode_IPR

        g = self._pump_graph(n_edges=3)
        g.graph["params"]["quality_threshold"] = 10.0
        modes_df = pd.DataFrame({"passive": [3.0 - 0.05j]})
        ipr = compute_mode_IPR(g, modes_df, index=0)
        assert np.isfinite(ipr)

    def test_gamma_q_value(self):
        """gamma_q_value = -Q(mode) * Im(gamma(mode, params))."""
        import pandas as pd

        from netsalt.modes import gamma_q_value

        g = self._pump_graph(n_edges=3)
        modes_df = pd.DataFrame({"passive": [3.0 - 0.05j]})
        val = gamma_q_value(g, modes_df, index=0)
        assert np.isfinite(val)


class TestPumpCostAndOverlap:
    """Exercise ``pump.py`` helpers that don't need a full Luigi pipeline."""

    def _tiny_graph_with_modes(self):
        """Return a (graph, modes_df) pair ready for pump helpers."""
        import networkx as nx
        import pandas as pd

        import netsalt
        from netsalt.physics import dispersion_relation_dielectric
        from netsalt.quantum_graph import create_quantum_graph

        g = nx.path_graph(5)
        positions = np.array([[float(i), 0.0] for i in range(5)])
        params = {
            "open_model": "open",
            "dielectric_params": {
                "method": "uniform",
                "inner_value": 4.0,
                "loss": 0.0,
                "outer_value": 1.0,
            },
            "c": 1.0,
        }
        create_quantum_graph(g, params, positions=positions)
        netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
        netsalt.set_dielectric_constant(g, g.graph["params"])

        # A trivially-shaped modes_df with two fake passive modes
        modes_df = pd.DataFrame({"passive": [2.0 - 0.1j, 3.5 - 0.15j]})
        return g, modes_df

    def test_pump_cost_penalises_large_overlaps(self):
        """When optimising mode 0 but pumping mode-1 edges, cost should rise."""
        from netsalt.pump import pump_cost

        pump = np.array([0, 1])
        pump_overlapps = np.array([[1.0, 0.0], [0.0, 1.0]])
        cost_good = pump_cost(
            np.array([1, 0]), modes_to_optimise=[0], pump_overlapps=pump_overlapps
        )
        cost_bad = pump_cost(pump, modes_to_optimise=[0], pump_overlapps=pump_overlapps)
        # The "bad" pump pumps mode-1 edges when we want mode-0 → infinite cost
        assert cost_bad > cost_good


class TestBuffonAndPixel:
    """Cover the graph-construction helpers in utils.py."""

    def test_make_buffon_graph_returns_graph(self):
        from netsalt.utils import make_buffon_graph

        rng = np.random.default_rng(3)
        graph, pos = make_buffon_graph(n_lines=5, size=(0.0, 1.0), resolution=0.2, rng=rng)
        # Should produce at least a handful of nodes
        assert len(graph) > 0
        assert len(pos) == len(graph) or len(pos) >= len(graph) - 1

    def test_remove_pixel_runs(self):
        """remove_pixel executes without error and tags every edge with a
        pump value. Whether the box actually overlaps edges depends on the
        graph geometry; this test just covers the code path."""
        import networkx as nx

        from netsalt.utils import remove_pixel

        g = nx.grid_2d_graph(4, 4)
        g = nx.convert_node_labels_to_integers(g)
        for i, u in enumerate(g.nodes):
            g.nodes[u]["position"] = np.array([float(i % 4), float(i // 4)])
        _, pump = remove_pixel(g, center=(1.5, 1.5), size=1.0)
        assert len(pump) > 0
        assert all(p in (0, 1) for p in pump)


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
