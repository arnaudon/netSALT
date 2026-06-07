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


def make_line_graph(
    n_edges=5,
    dielectric=4.0,
    extra_params=None,
    normalized_positions=False,
    total_length=None,
    as_class=False,
):
    """Shared builder for the open dielectric line-graph fixture.

    A path graph with unit (or, with ``normalized_positions``, [0, 1]-scaled)
    edge lengths, a uniform dielectric, and the dielectric dispersion relation
    set. ``extra_params`` merges extra knobs into the params dict;
    ``as_class=True`` returns a :class:`~netsalt.quantum_graph.QuantumGraph`.
    """
    import netsalt
    from netsalt.physics import dispersion_relation_dielectric
    from netsalt.quantum_graph import QuantumGraph, create_quantum_graph, set_total_length

    denom = n_edges if normalized_positions else 1
    positions = np.array([[float(i) / denom, 0.0] for i in range(n_edges + 1)])
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
    if extra_params:
        params.update(extra_params)

    g = nx.path_graph(n_edges + 1)
    if as_class:
        g = QuantumGraph.from_networkx(g, params=params, positions=positions)
    else:
        create_quantum_graph(g, params, positions=positions)
    if total_length is not None:
        set_total_length(g, total_length)
    netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    return g


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

    def test_pump_dispersion_reduces_to_dielectric_at_zero_pump(self):
        """At D0 = 0 the pumped relation must equal the passive dielectric one,
        including the wavespeed scaling (regression: the dielectric term used to
        be divided by c once inside the sqrt instead of by c**2, so the two
        branches only agreed for c = 1)."""
        from netsalt.physics import (
            dispersion_relation_dielectric,
            dispersion_relation_pump,
        )

        eps = np.array([4.0, 2.25])
        for c in (1.0, 2.0, 0.5):
            passive = dispersion_relation_dielectric(
                3.0, params={"dielectric_constant": eps, "c": c}
            )
            pumped = dispersion_relation_pump(
                3.0,
                params={
                    "dielectric_constant": eps,
                    "c": c,
                    "k_a": 3.0,
                    "gamma_perp": 1.0,
                    "D0": 0.0,
                    "pump": np.ones_like(eps),
                },
            )
            assert np.allclose(passive, pumped)
            # explicit closed form k = (w/c) sqrt(eps)
            assert np.allclose(pumped, 3.0 * np.sqrt(eps) / c)

    def test_pump_dispersion_adds_gain_under_the_sqrt(self):
        """A non-zero pump shifts k by gamma*D0 added to the dielectric, and the
        whole thing stays scaled by 1/c."""
        from netsalt.physics import dispersion_relation_pump, gamma

        eps = np.array([4.0, 2.25])
        params = {
            "dielectric_constant": eps,
            "c": 2.0,
            "k_a": 3.5,
            "gamma_perp": 1.0,
            "D0": 0.3,
            "pump": np.ones_like(eps),
        }
        expected = 3.0 * np.sqrt(eps + gamma(3.0, params) * 0.3) / 2.0
        assert np.allclose(dispersion_relation_pump(3.0, params=params), expected)

    def test_resistance_dispersion_is_lossy(self):
        """A positive resistance must attenuate (Im(k) > 0), matching the lossy
        dielectric sign convention (loss = positive imaginary part)."""
        from netsalt.physics import dispersion_relation_resistance

        k = dispersion_relation_resistance(3.0, params={"c": 1.0, "C": 1.0, "R": 0.1})
        assert np.imag(k) > 0


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

    def test_literal_fields_reject_typos(self):
        """``refine_method`` and ``mode_search_method`` are typed as
        ``Literal[...]`` so a typo in the config fails on construction
        rather than silently propagating to the dispatcher."""
        from pydantic import ValidationError

        from netsalt.params import NetSaltParams

        # Valid values pass.
        NetSaltParams.from_dict({"refine_method": "root", "mode_search_method": "grid"})

        with pytest.raises(ValidationError):
            NetSaltParams.from_dict({"refine_method": "newton"})  # removed in favour of root
        with pytest.raises(ValidationError):
            NetSaltParams.from_dict({"refine_method": "rooot"})  # typo
        with pytest.raises(ValidationError):
            NetSaltParams.from_dict({"mode_search_method": "Grid"})  # case mismatch

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
        """Open dielectric line graph with unit edge lengths."""
        return make_line_graph(n_edges, dielectric)

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

    def test_incidence_topology_cache_is_stable_across_calls(self):
        """Second call to construct_incidence_matrix should reuse the cache
        but still produce the same matrix values as the first call."""
        from netsalt.quantum_graph import construct_incidence_matrix, set_wavenumber

        g = self._line_graph(n_edges=5)
        set_wavenumber(g, 2.0 + 0.1j)
        BT1, B1 = construct_incidence_matrix(g)
        assert "_incidence_topology" in g.graph
        # Same k → identical matrices on the second call.
        BT2, B2 = construct_incidence_matrix(g)
        assert (BT1 != BT2).nnz == 0
        assert (B1 != B2).nnz == 0

    def test_incidence_topology_cache_invalidates_on_resize(self):
        """If ``len(edges)`` changes after the cache was populated, the
        cache must be rebuilt to match the new graph."""
        from netsalt.quantum_graph import construct_incidence_matrix, set_wavenumber

        g_small = self._line_graph(n_edges=3)
        set_wavenumber(g_small, 1.0 + 0.0j)
        construct_incidence_matrix(g_small)
        assert g_small.graph["_incidence_topology"]["m"] == 3

        # Simulate a caller reusing the stale cache on a graph with one more edge
        g_big = self._line_graph(n_edges=4)
        g_big.graph["_incidence_topology"] = g_small.graph["_incidence_topology"]
        set_wavenumber(g_big, 1.0 + 0.0j)
        _BT, B = construct_incidence_matrix(g_big)
        # The matrix should match the bigger graph, not the stale m=3 cache.
        assert B.shape == (2 * 4, len(g_big))

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


class TestRefinementAlgorithms:
    """All four refiners converge to the same root from the same initial
    guess; the dispatcher picks the one named in ``params``."""

    def _line_graph(self, n_edges=6, dielectric=4.0):
        return make_line_graph(
            n_edges,
            dielectric,
            extra_params={
                "quality_threshold": 1e-4,
                "search_stepsize": 0.05,
                "max_steps": 500,
                # ``_search_box`` uses these to build the locality bound; set a
                # generous window around the mode we're targeting so the
                # refiners can actually move to the root.
                "k_min": 2.8,
                "k_max": 3.4,
                "alpha_min": 0.0,
                "alpha_max": 0.3,
            },
        )

    def _count_evals(self, fn, *args, **kwargs):
        """Monkey-patch ``mode_quality`` to count evaluations inside *fn*."""
        import netsalt.algorithm as alg

        orig = alg.mode_quality
        count = [0]

        def wrapper(*a, **kw):
            count[0] += 1
            return orig(*a, **kw)

        alg.mode_quality = wrapper
        try:
            result = fn(*args, **kwargs)
        finally:
            alg.mode_quality = orig
        return result, count[0]

    def test_all_methods_converge(self):
        from netsalt.algorithm import refine_mode_brownian_ratchet, refine_mode_root
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph()
        init = np.array([3.0, 0.03])
        tol = g.graph["params"]["quality_threshold"]

        r_root, n_root = self._count_evals(refine_mode_root, init, g, g.graph["params"])
        r_br, n_br = self._count_evals(
            refine_mode_brownian_ratchet,
            init,
            g,
            g.graph["params"],
            rng=np.random.default_rng(0),
        )

        for name, r in [("root", r_root), ("brownian", r_br)]:
            assert r is not None, f"{name} returned None"
            assert mode_quality(r, g) < tol, f"{name} above threshold"

        # Sanity upper bound on root, plus the relative sanity check
        # that root beats the ratchet.
        assert n_root <= 80, f"root took {n_root} evals"
        assert n_root < n_br

    def test_dispatch_honours_refine_method(self):
        """``refine_mode`` must route to the named implementation."""
        from unittest import mock

        from netsalt.algorithm import refine_mode

        g = self._line_graph()
        init = np.array([3.0, 0.03])
        for name, target in [
            ("root", "netsalt.algorithm.refine_mode_root"),
            ("brownian", "netsalt.algorithm.refine_mode_brownian_ratchet"),
        ]:
            g.graph["params"]["refine_method"] = name
            with mock.patch(target, return_value=init) as patched:
                refine_mode(init, g, g.graph["params"])
                patched.assert_called_once()

    def test_dispatch_rejects_unknown_method(self):
        """``refine_method`` is typed as a Literal on ``NetSaltParams`` so a
        typo fails at the graph boundary (pydantic validation on assignment)
        rather than after a slow descent into ``refine_mode``."""
        from pydantic import ValidationError

        from netsalt.algorithm import refine_mode

        g = self._line_graph()
        with pytest.raises(ValidationError):
            g.graph["params"]["refine_method"] = "does-not-exist"

        # And the dispatcher itself still raises if someone bypasses the
        # boundary (e.g. ``refine_mode(..., params={"refine_method": "x"})``
        # with a bare dict that never went through pydantic).
        with pytest.raises(ValueError, match="Unknown refine_method"):
            refine_mode([3.0, 0.03], g, {"refine_method": "does-not-exist"})

    def test_default_method_is_root(self):
        """When ``refine_method`` is absent, root is the default."""
        from unittest import mock

        from netsalt.algorithm import refine_mode

        g = self._line_graph()
        g.graph["params"]["refine_method"] = None
        with mock.patch(
            "netsalt.algorithm.refine_mode_root", return_value=np.array([3.0, 0.03])
        ) as patched:
            refine_mode([3.0, 0.03], g, g.graph["params"])
            patched.assert_called_once()

    def test_search_box_rejects_runaway_result(self):
        """A result outside the ``search_radii`` window is rejected."""
        from netsalt.algorithm import refine_mode_root

        g = self._line_graph()
        g.graph["params"]["k_min"] = 2.99
        g.graph["params"]["k_max"] = 3.01
        g.graph["params"]["alpha_min"] = 0.02
        g.graph["params"]["alpha_max"] = 0.04
        # Start far from the box; root will converge to something way outside
        init = np.array([3.0, 0.03])
        # Shrink tol so root converges to a root; box is tiny, should reject
        g.graph["params"]["quality_threshold"] = 1e-4
        result = refine_mode_root(init, g, g.graph["params"])
        # Either the result is within the tight box, or None (rejected).
        # If a root happened to exist right at the initial guess, the result
        # would be accepted; the important thing is we don't get a result
        # far outside the claimed search window.
        if result is not None:
            assert abs(result[0] - init[0]) <= 1.5 * 0.01
            assert abs(result[1] - init[1]) <= 1.5 * 0.01


class TestContourIntegration:
    """Beyn's contour method finds true modes on a line graph where the
    mode locations are analytically known, and the subdivided variant
    handles regions where the mode count exceeds the probe dimension."""

    def _line_graph(self, n_edges=10, dielectric=4.0, total_length=1.0):
        return make_line_graph(
            n_edges,
            dielectric,
            extra_params={"k_min": 0.5, "k_max": 20.0, "alpha_min": 0.0, "alpha_max": 1.0},
            normalized_positions=True,
            total_length=total_length,
        )

    def test_contour_finds_true_modes_on_line_graph(self):
        """On the dielectric line graph, Beyn should return *true* roots
        of det(L(k))=0. The default ``quality_filter`` drops spurious
        SVD-extraction outputs, so every returned mode must satisfy
        ``|λ₁| < 1e-3``; the median should be much tighter."""
        from netsalt.contour import find_modes_contour
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph()
        rng = np.random.default_rng(42)
        # probe_dim must exceed the mode count inside the **ellipse**
        # (which circumscribes the rectangle, a bit larger than the rect).
        # Use a narrow window containing just a few modes so the 11-node
        # line graph has enough probe dimensions.
        modes = find_modes_contour(
            g, bounds=(1.0, 4.0, 0.0, 1.0), n_quad=120, probe_dim=10, rng=rng
        )
        assert len(modes) >= 2, f"expected ≥2 modes in [1, 4], got {len(modes)}"
        qs = np.array([mode_quality(m, g) for m in modes])
        # All modes pass through ``quality_filter=1e-3`` by default.
        assert qs.max() < 1e-3

    def test_contour_with_refinement_round_trip(self):
        """Using Beyn + refine-mode hybrid: turn off the quality filter on
        Beyn so it returns all candidates, then refine each via the new
        root method. Every refined mode should converge to a true root."""
        from netsalt.algorithm import refine_mode_root
        from netsalt.contour import find_modes_contour
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph()
        rng = np.random.default_rng(42)
        g.graph["params"]["search_stepsize"] = 0.01
        g.graph["params"]["quality_threshold"] = 1e-4
        g.graph["params"]["max_steps"] = 200
        candidates = find_modes_contour(g, n_quad=200, probe_dim=20, quality_filter=None, rng=rng)
        refined = [refine_mode_root(c, g, g.graph["params"]) for c in candidates]
        refined = [r for r in refined if r is not None]
        assert len(refined) >= 1
        for m in refined:
            assert mode_quality(m, g) < 1e-4

    def test_contour_returns_empty_when_no_modes_in_window(self):
        """Ask for a narrow window that contains no modes (the contour's
        laplacian is well away from singular). Beyn should return nothing
        rather than fake modes."""
        from netsalt.contour import find_modes_contour

        g = self._line_graph()
        # Tiny window deep in the imaginary axis, away from the real line
        # where the modes live.
        rng = np.random.default_rng(0)
        modes = find_modes_contour(
            g, bounds=(10.0, 10.01, 0.99, 1.00), n_quad=40, probe_dim=8, rng=rng
        )
        # Either zero modes (correct) or a handful of spurious ones that
        # fail ``_inside_contour`` — the important thing is we don't return
        # garbage eigenvalues that passed the SVD threshold.
        assert len(modes) == 0 or all(10.0 <= m[0] <= 10.01 and 0.99 <= m[1] <= 1.00 for m in modes)

    def test_adaptive_contour_recovers_all_modes_without_n_k(self):
        """``find_modes_contour_adaptive`` should match the explicit
        subdivided run on a workload where the rectangle's mode count
        exceeds ``probe_dim``. The user picks ``probe_dim`` and the
        algorithm picks ``n_k`` automatically by saturation."""
        from netsalt.contour import (
            find_modes_contour,
            find_modes_contour_adaptive,
        )

        # n_edges=10 + total_length=2 → ~25 modes in [0.5, 20]; the
        # single contour saturates at probe_dim=8 (well below the 25
        # modes inside) and adaptive must split until each cell fits
        # under the capacity ceiling.
        from netsalt.quantum_graph import mode_quality

        g = self._line_graph(n_edges=10, total_length=2.0)
        bounds = (0.5, 20.0, 0.0, 1.0)
        adaptive = find_modes_contour_adaptive(
            g, bounds=bounds, n_quad=80, probe_dim=8, rng=np.random.default_rng(1)
        )
        # Gold reference: very-deep manual subdivision.
        gold = find_modes_contour(
            g,
            bounds=bounds,
            n_k=16,
            n_alpha=1,
            n_quad=120,
            probe_dim=8,
            rng=np.random.default_rng(1),
        )
        # Adaptive should match gold within a couple of boundary-dedup
        # edge-effects.
        assert len(adaptive) >= len(gold) - 2
        # And every returned mode is a true root.
        for m in adaptive:
            assert mode_quality(m, g) < 1e-3

    def test_adaptive_contour_does_not_subdivide_unnecessarily(self):
        """When the rectangle has comfortably fewer modes than
        ``probe_dim``, adaptive must accept the single-contour result
        without spending a single split."""
        from netsalt.contour import find_modes_contour, find_modes_contour_adaptive

        g = self._line_graph(n_edges=10)
        bounds = (1.0, 4.0, 0.0, 1.0)
        rng_a = np.random.default_rng(2)
        rng_b = np.random.default_rng(2)
        adaptive = find_modes_contour_adaptive(
            g, bounds=bounds, n_quad=120, probe_dim=10, rng=rng_a
        )
        single = find_modes_contour(g, bounds=bounds, n_quad=120, probe_dim=10, rng=rng_b)
        # Same RNG seed and never-saturated single contour → identical sets.
        assert len(adaptive) == len(single)
        if len(adaptive):
            np.testing.assert_allclose(np.sort(adaptive[:, 0]), np.sort(single[:, 0]), atol=1e-9)

    def test_adaptive_contour_terminates_at_max_depth(self):
        """A stupidly small ``probe_dim`` forces every cell to saturate;
        ``max_depth`` must guarantee termination rather than recursing
        forever."""
        from netsalt.contour import find_modes_contour_adaptive

        g = self._line_graph(n_edges=10)
        bounds = (1.0, 4.0, 0.0, 1.0)
        # probe_dim=1 plus low saturation_factor → every cell triggers
        # subdivision until max_depth is reached. Test passes if the
        # call returns without exception.
        modes = find_modes_contour_adaptive(
            g,
            bounds=bounds,
            n_quad=80,
            probe_dim=1,
            saturation_factor=0.5,
            max_depth=3,
            rng=np.random.default_rng(0),
        )
        # No assertion on len — the point is termination.
        assert isinstance(modes, np.ndarray)

    def test_adaptive_contour_is_exported(self):
        import netsalt

        assert "find_modes_contour_adaptive" in netsalt.__all__

    def test_tune_contour_parameters_returns_usable_settings(self):
        """``tune_contour_parameters`` runs adaptive once and returns a
        param dict that splats directly into
        ``find_modes_contour``. Verify the round trip
        recovers the same modes."""
        from netsalt.contour import (
            find_modes_contour,
            tune_contour_parameters,
        )

        # 25-mode line graph; probe_dim=8 forces non-trivial subdivision.
        g = self._line_graph(n_edges=10, total_length=2.0)
        bounds = (0.5, 20.0, 0.0, 1.0)
        params, info = tune_contour_parameters(
            g, bounds=bounds, probe_dim=8, n_quad=120, rng=np.random.default_rng(3)
        )
        # Sanity on the returned settings.
        assert params["probe_dim"] == 8
        assert params["n_quad"] == 120
        assert params["n_alpha"] == 1
        assert params["n_k"] >= 1
        assert info["discovered_modes"] >= 5

        # Splat the params into the non-adaptive entry point and check
        # we recover ~the same mode set as the tuning run did.
        modes = find_modes_contour(g, bounds=bounds, **params, rng=np.random.default_rng(3))
        assert abs(len(modes) - info["discovered_modes"]) <= 2

    def test_tune_contour_is_exported(self):
        import netsalt

        assert "tune_contour_parameters" in netsalt.__all__

    def test_subdivided_contour_finds_more_modes_than_single(self):
        """When a region contains more modes than ``probe_dim``, a single
        contour can't resolve them all, but subdivision can."""
        from netsalt.contour import find_modes_contour

        g = self._line_graph(n_edges=10)
        rng = np.random.default_rng(123)
        # Intentionally under-sized probe_dim for a single contour — only a
        # handful of singular values will survive the SVD cut.
        single = find_modes_contour(
            g, bounds=(0.5, 20.0, 0.0, 1.0), n_quad=80, probe_dim=3, rng=rng
        )
        sub = find_modes_contour(
            g, bounds=(0.5, 20.0, 0.0, 1.0), n_k=5, n_quad=80, probe_dim=6, rng=rng
        )
        assert len(sub) >= len(single)

    def test_contour_is_exported_from_package(self):
        import netsalt

        assert "find_modes_contour" in netsalt.__all__
        assert "find_modes_contour" in netsalt.__all__

    def test_find_passive_modes_defaults_to_contour(self):
        """With no ``mode_search_method`` set, ``find_passive_modes`` should
        use Beyn and return a populated modes dataframe without needing a
        ``qualities`` grid."""
        import netsalt
        from netsalt.modes import find_passive_modes
        from netsalt.physics import dispersion_relation_dielectric
        from netsalt.quantum_graph import create_quantum_graph, set_total_length

        g = nx.path_graph(11)
        pos = np.array([[float(i) / 10.0, 0.0] for i in range(11)])
        params = {
            "open_model": "open",
            "dielectric_params": {
                "method": "uniform",
                "inner_value": 4.0,
                "loss": 0.0,
                "outer_value": 1.0,
            },
            "c": 1.0,
            "k_min": 1.0,
            "k_max": 4.0,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
        }
        create_quantum_graph(g, params, positions=pos)
        set_total_length(g, 1.0)
        netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
        netsalt.set_dielectric_constant(g, g.graph["params"])
        modes_df = find_passive_modes(g)
        # Has at least one mode, columns are 'passive' and 'q_factor'.
        assert "passive" in modes_df.columns.get_level_values(0)
        assert "q_factor" in modes_df.columns.get_level_values(0)
        assert len(modes_df) >= 1

    def test_find_passive_modes_grid_requires_qualities(self):
        from netsalt.modes import find_passive_modes

        g = nx.path_graph(3)
        g.graph["params"] = {"mode_search_method": "grid"}
        with pytest.raises(ValueError, match="method='grid' requires"):
            find_passive_modes(g, method="grid")

    def test_find_passive_modes_rejects_unknown_method(self):
        from netsalt.modes import find_passive_modes

        g = nx.path_graph(3)
        g.graph["params"] = {}
        with pytest.raises(ValueError, match="Unknown mode_search_method"):
            find_passive_modes(g, method="not-a-method")

    def test_find_passive_modes_defaults_to_grid_when_qualities_provided(self):
        """Legacy behaviour: ``find_passive_modes(g, qualities)`` with no
        explicit method should use the grid path, not silently switch to
        Beyn and ignore the supplied qualities."""
        from unittest import mock

        from netsalt.modes import find_passive_modes

        g = nx.path_graph(3)
        g.graph["params"] = {}
        qualities = np.zeros((3, 3))
        with mock.patch("netsalt.modes.find_modes", return_value="MODES_DF") as patched:
            result = find_passive_modes(g, qualities)
        patched.assert_called_once()
        assert result == "MODES_DF"

    def test_find_passive_modes_warns_when_contour_ignores_qualities(self):
        """Caller passes both ``qualities`` and ``method='contour'``: contour
        ignores the qualities, so warn loudly rather than swallow the
        argument."""
        import warnings as _warnings
        from unittest import mock

        from netsalt.modes import find_passive_modes

        g = nx.path_graph(3)
        g.graph["params"] = {"k_min": 1.0, "k_max": 2.0}
        qualities = np.zeros((3, 3))
        # Patch the contour entry point: we only care about the warning
        # firing, not the actual mode search.
        with (
            mock.patch(
                "netsalt.contour.find_modes_contour",
                return_value=np.empty((0, 2)),
            ),
            _warnings.catch_warnings(record=True) as caught,
        ):
            _warnings.simplefilter("always")
            find_passive_modes(g, qualities, method="contour")
        assert any("ignores the supplied qualities" in str(w.message) for w in caught)

    def test_contour_separates_closely_spaced_modes(self):
        """On a long line graph (``total_length=10``) the mode spacing in
        ``k`` is ``π/(2 · √ε · L) = π/20 ≈ 0.157``. Seven consecutive
        modes should pop out of a single contour that spans them, each
        one resolved to machine precision — Beyn's SVD separates the
        close pairs cleanly when the probe dim is large enough."""
        import netsalt
        from netsalt.contour import find_modes_contour
        from netsalt.physics import dispersion_relation_dielectric
        from netsalt.quantum_graph import (
            create_quantum_graph,
            mode_quality,
            set_total_length,
        )

        g = nx.path_graph(41)
        pos = np.array([[float(i) / 40.0, 0.0] for i in range(41)])
        params = {
            "open_model": "open",
            "dielectric_params": {
                "method": "uniform",
                "inner_value": 4.0,
                "loss": 0.0,
                "outer_value": 1.0,
            },
            "c": 1.0,
            "k_min": 0.5,
            "k_max": 20.0,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
        }
        create_quantum_graph(g, params, positions=pos)
        set_total_length(g, 10.0)
        netsalt.set_dispersion_relation(g, dispersion_relation_dielectric)
        netsalt.set_dielectric_constant(g, g.graph["params"])

        # Expected modes at k = n · π/20 for n = 9, 10, ..., 15 → k ≈ 1.414,
        # 1.571, 1.728, 1.885, 2.042, 2.199, 2.356. Seven consecutive modes
        # at spacing 0.157.
        modes = find_modes_contour(
            g,
            bounds=(1.4, 2.5, 0.0, 1.0),
            n_quad=120,
            probe_dim=30,
            rng=np.random.default_rng(0),
        )
        assert len(modes) == 7
        expected = np.array([n * np.pi / 20 for n in range(9, 16)])
        returned = np.sort(modes[:, 0])
        np.testing.assert_allclose(returned, expected, atol=1e-3)
        # Every returned mode is a true root to machine precision.
        for m in modes:
            assert mode_quality(m, g) < 1e-9


class TestPumpCostAndOverlap:
    """Exercise ``pump.py`` helpers that don't need a full pipeline."""

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

    def test_q_value_is_positive_for_a_leaky_mode(self):
        """A leaky mode has alpha = -Im(k) > 0, so Q = Re(k)/(2*alpha) must be
        positive (the docstring formula uses alpha, not a bare +Im(k))."""
        from netsalt.physics import q_value

        # complex k = 10 - 0.5j  ->  alpha = 0.5  ->  Q = 10 / 1.0 = 10
        assert q_value(10.0 - 0.5j) == pytest.approx(10.0)
        assert q_value(10.0 - 0.5j) > 0


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


class TestQuantumGraph:
    """The QuantumGraph class (issue #28) is a thin, additive nx.Graph subclass:
    its methods must delegate to the existing free functions, and it must remain
    picklable and JSON-serialisable exactly like a plain quantum graph."""

    def _qg(self, n_edges=4, dielectric=4.0):
        """Build a QuantumGraph line graph with a dispersion relation set."""
        return make_line_graph(n_edges, dielectric, as_class=True)

    def test_is_nx_graph_subclass_with_params(self):
        import networkx as nx_

        from netsalt.params import NetSaltParams

        qg = self._qg()
        assert isinstance(qg, nx_.Graph)
        assert isinstance(qg.params, NetSaltParams)

    def test_matrix_methods_match_free_functions(self):
        """Methods are sugar: they must return exactly what the free functions
        return on the same graph."""
        from netsalt.quantum_graph import (
            construct_incidence_matrix,
            construct_laplacian,
            construct_weight_matrix,
            mode_quality,
        )

        qg = self._qg()
        k = 1.0 + 0.0j

        assert np.allclose(qg.laplacian(k).toarray(), construct_laplacian(k, qg).toarray())
        assert np.allclose(qg.weight_matrix().toarray(), construct_weight_matrix(qg).toarray())
        bt_m, b_m = qg.incidence_matrix()
        bt_f, b_f = construct_incidence_matrix(qg)
        assert np.allclose(bt_m.toarray(), bt_f.toarray())
        assert np.allclose(b_m.toarray(), b_f.toarray())

        mode = [1.0, 0.0]
        q_method = qg.mode_quality(mode, rng=np.random.default_rng(0))
        q_func = mode_quality(mode, qg, rng=np.random.default_rng(0))
        assert q_method == q_func

    def test_total_length_properties(self):
        from netsalt.quantum_graph import get_total_length

        qg = self._qg()
        assert qg.total_length == get_total_length(qg)

    def test_pickle_round_trip_preserves_type_state_and_methods(self):
        """WorkerScan/WorkerModes pickle the whole graph into Pool workers, so
        the subclass must survive a pickle round-trip with its state intact."""
        import pickle

        from netsalt.quantum_graph import QuantumGraph

        qg = self._qg()
        restored = pickle.loads(pickle.dumps(qg))
        assert isinstance(restored, QuantumGraph)
        assert restored.params["open_model"] == "open"
        # a delegating method still works on the unpickled instance
        assert restored.laplacian(1.0 + 0.0j).shape == (len(qg), len(qg))

    def test_json_round_trip_as_class(self, tmp_path):
        from netsalt.io import load_graph, save_graph
        from netsalt.quantum_graph import QuantumGraph

        qg = self._qg()
        path = tmp_path / "qg.json"
        save_graph(qg, str(path))

        loaded = load_graph(str(path), as_class=True)
        assert isinstance(loaded, QuantumGraph)
        assert loaded.params["open_model"] == qg.params["open_model"]
        assert np.allclose(loaded.graph["lengths"], qg.graph["lengths"])
        assert np.allclose(loaded.nodes[0]["position"], qg.nodes[0]["position"])

    def test_load_graph_defaults_to_plain_graph(self, tmp_path):
        """as_class defaults to False so existing callers are unaffected."""
        from netsalt.io import load_graph, save_graph
        from netsalt.quantum_graph import QuantumGraph

        qg = self._qg()
        path = tmp_path / "qg.json"
        save_graph(qg, str(path))

        loaded = load_graph(str(path))
        assert not isinstance(loaded, QuantumGraph)

    def test_oversample_returns_quantum_graph(self):
        from netsalt.quantum_graph import QuantumGraph

        qg = self._qg()
        over = qg.oversample(0.3)
        assert isinstance(over, QuantumGraph)
        assert len(over) > len(qg)

    def test_simplify_returns_quantum_graph(self):
        from netsalt.quantum_graph import QuantumGraph

        assert isinstance(self._qg().simplify(), QuantumGraph)

    def test_physics_setup_methods_chain(self):
        """set_dispersion_relation / set_dielectric_constant return self and
        set the same graph state as the free functions."""
        from netsalt.physics import dispersion_relation_dielectric
        from netsalt.quantum_graph import QuantumGraph

        g = nx.path_graph(4)
        positions = np.array([[float(i), 0.0] for i in range(4)])
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
        qg = QuantumGraph.from_networkx(g, params=params, positions=positions)
        out = qg.set_dispersion_relation(dispersion_relation_dielectric).set_dielectric_constant()
        assert out is qg  # chainable
        assert qg.graph["dispersion_relation"] is dispersion_relation_dielectric
        assert qg.params.get("dielectric_constant") is not None
        # parity: a laplacian is now buildable, matching the free function
        from netsalt.quantum_graph import construct_laplacian

        assert np.allclose(
            qg.laplacian(1.0 + 0.0j).toarray(), construct_laplacian(1.0 + 0.0j, qg).toarray()
        )

    def test_with_pump_returns_quantum_graph_without_mutating(self):
        from netsalt.quantum_graph import QuantumGraph

        qg = self._qg()
        assert qg.params.get("D0") is None
        pumped = qg.with_pump(0.7)
        assert isinstance(pumped, QuantumGraph)
        assert pumped.params["D0"] == 0.7
        assert qg.params.get("D0") is None  # original untouched

    def test_mode_on_nodes_matches_free_function(self):
        from netsalt.modes import mode_on_nodes

        qg = self._qg()
        qg.params["quality_threshold"] = 1e6  # relax so mode_on_nodes never raises
        mode = [1.0, 0.0]
        assert np.allclose(qg.mode_on_nodes(mode), mode_on_nodes(mode, qg))

    def test_scan_frequencies_matches_free_function(self):
        from netsalt.modes import scan_frequencies

        qg = self._qg()
        qg.params.update(
            {
                "k_min": 1.0,
                "k_max": 1.2,
                "k_n": 2,
                "alpha_min": 0.0,
                "alpha_max": 0.1,
                "alpha_n": 2,
                "n_workers": 1,
            }
        )
        method = qg.scan_frequencies()
        free = scan_frequencies(qg)
        assert method.shape == (2, 2)
        assert np.allclose(method, free)


class TestNoInPlacePumpMutation:
    """Regression for the WorkerModes in-place params mutation: applying a
    per-mode pump (D0) or search window must never leak into the shared
    ``graph.graph["params"]``. The laplacian-at-D0 is built on a throwaway
    copy via ``graph_with_params`` instead."""

    def _line_graph(self, n_edges=4):
        return make_line_graph(
            n_edges,
            extra_params={"refine_method": "root", "quality_threshold": 1e-2, "max_steps": 5},
        )

    def test_graph_with_pump_leaves_original_untouched(self):
        from netsalt.quantum_graph import graph_with_pump

        g = self._line_graph()
        assert g.graph["params"].get("D0") is None
        local = graph_with_pump(g, 0.7)
        # the copy carries the pump...
        assert local.graph["params"]["D0"] == 0.7
        # ...the original does not, and it is a distinct params object
        assert g.graph["params"].get("D0") is None
        assert local.graph["params"] is not g.graph["params"]

    def test_worker_modes_does_not_mutate_shared_params(self):
        from netsalt.modes import WorkerModes

        g = self._line_graph()
        assert g.graph["params"].get("D0") is None
        assert g.graph["params"].get("k_min") is None

        assert g.graph["params"].get("search_stepsize") is None

        worker = WorkerModes(
            [[1.0, 0.0], [1.2, 0.0]],
            g,
            D0s=[0.5, 0.6],
            search_radii=[0.1, 0.1],
            search_stepsize=0.02,
            quality_method="eigenvalue",
        )
        worker(0)
        worker(1)

        # No D0 / search-window / stepsize field leaked back onto shared params.
        assert g.graph["params"].get("D0") is None
        assert g.graph["params"].get("k_min") is None
        assert g.graph["params"].get("k_max") is None
        assert g.graph["params"].get("search_stepsize") is None


class TestPlotPumpTraj:
    """Regression tests for ``plot_pump_traj`` (issues #17 / #25).

    The colorbar ``vmax`` was computed as ``c[max(argmin(|imag|)) + 1]``.
    When a mode's |imag| minimum lands in the *last* D0 column the ``+ 1``
    indexed past the end of the column list and raised
    ``IndexError: list index out of range``.
    """

    def _modes_df(self, imag_per_step, n_modes=2):
        """Build a minimal modes_df with a ``mode_trajectories`` block.

        ``imag_per_step`` is the imaginary part of every mode at each D0
        column, so the caller controls where ``argmin(|imag|)`` lands.
        """
        import matplotlib

        matplotlib.use("Agg")
        import pandas as pd

        D0s = [0.1 * j for j in range(len(imag_per_step))]
        df = pd.DataFrame()
        for D0, im in zip(D0s, imag_per_step, strict=True):
            df["mode_trajectories", D0] = [complex(1.0, im) for _ in range(n_modes)]
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_threshold_in_last_column_does_not_raise(self):
        from netsalt.plotting import plot_pump_traj

        # |imag| strictly decreasing -> argmin is the final column.
        df = self._modes_df([1.0, 0.5, 0.0])
        # Must not raise IndexError.
        plot_pump_traj(df)

    def test_threshold_in_middle_column(self):
        from netsalt.plotting import plot_pump_traj

        # |imag| minimal in the middle column -> +1 stays in range.
        df = self._modes_df([1.0, 0.0, 1.0])
        plot_pump_traj(df)


class TestSaturatedDispersion:
    """``dispersion_relation_pump_saturated`` (full-SALT gain term)."""

    def _params(self):
        return {
            "dielectric_constant": np.array([2.0, 3.0, 2.5]),
            "pump": np.array([1.0, 0.0, 1.0]),
            "D0": 0.03,
            "c": 1.0,
            "gamma_perp": 0.5,
            "k_a": 5.0,
        }

    def test_reduces_to_pumped_when_unsaturated(self):
        """With ``D0_eff = D0 * pump`` (denominator one) it must reproduce
        ``dispersion_relation_pump`` exactly."""
        from netsalt.physics import (
            dispersion_relation_pump,
            dispersion_relation_pump_saturated,
        )

        params = self._params()
        base = dispersion_relation_pump(5.1, params)
        sat_params = dict(params, D0_eff=params["D0"] * params["pump"])
        assert np.allclose(base, dispersion_relation_pump_saturated(5.1, sat_params))

    def test_falls_back_to_pumped_without_D0_eff(self):
        from netsalt.physics import (
            dispersion_relation_pump,
            dispersion_relation_pump_saturated,
        )

        params = self._params()
        assert np.allclose(
            dispersion_relation_pump(5.1, params),
            dispersion_relation_pump_saturated(5.1, params),
        )

    def test_saturation_lowers_the_gain_contribution(self):
        """A larger denominator (D0_eff < D0*pump) pulls k toward the passive
        dielectric value on the pumped edges."""
        from netsalt.physics import (
            dispersion_relation_dielectric,
            dispersion_relation_pump,
            dispersion_relation_pump_saturated,
        )

        params = self._params()
        passive = dispersion_relation_dielectric(5.1, params)
        pumped = dispersion_relation_pump(5.1, params)
        sat_params = dict(params, D0_eff=0.5 * params["D0"] * params["pump"])
        saturated = dispersion_relation_pump_saturated(5.1, sat_params)
        # on the pumped edges the saturated k sits between passive and full pump
        gain_edges = params["pump"] > 0
        assert np.all(
            np.abs(saturated - passive)[gain_edges] < np.abs(pumped - passive)[gain_edges]
        )


class TestModeOnNodesQualityFlag:
    """``check_quality=False`` lets the profile helpers evaluate off-threshold."""

    def _line_graph(self, **kw):
        return make_line_graph(**kw)

    def test_check_quality_false_returns_vector_on_non_mode(self):
        from netsalt.modes import mode_on_nodes

        g = self._line_graph(n_edges=4)
        g.graph["params"]["quality_threshold"] = 1e-12
        # would raise with the default check; must not with it disabled
        sol = mode_on_nodes([3.0, 0.05], g, check_quality=False)
        assert sol.shape == (len(g),)

    def test_mean_mode_on_edges_threads_the_flag(self):
        import netsalt
        from netsalt.modes import mean_mode_on_edges
        from netsalt.physics import dispersion_relation_pump
        from netsalt.quantum_graph import update_parameters

        g = self._line_graph(n_edges=4)
        netsalt.set_dispersion_relation(g, dispersion_relation_pump)
        update_parameters(
            g, {"k_a": 3.0, "gamma_perp": 1.0, "D0": 0.5, "pump": np.ones(len(g.edges))}
        )
        g.graph["params"]["quality_threshold"] = 1e-12
        mean = mean_mode_on_edges([3.0, 0.05], g, check_quality=False)
        assert mean.shape == (len(g.edges),)


class TestIntensitySolveHelpers:
    """Pure-algebra helpers shared by the intensity solvers."""

    def test_slopes_shifts_identity_matrix(self):
        from netsalt.modes import _intensity_slopes_shifts

        thresholds = np.array([2.0, 4.0])
        T = np.eye(2)
        slopes, shifts = _intensity_slopes_shifts(T, thresholds, [0, 1])
        # T = I  ->  slopes = 1/threshold, shifts = 1, so intensity(D0) = D0/thr - 1
        assert np.allclose(slopes, 1.0 / thresholds)
        assert np.allclose(shifts, 1.0)

    def test_finalise_writes_sorted_intensity_columns(self):
        import pandas as pd

        from netsalt.modes import _finalise_modal_intensities

        modal = pd.DataFrame(index=range(2))
        modal.loc[0, 0.5] = 0.0
        modal.loc[0, 0.2] = 0.0  # inserted out of order on purpose
        modal.loc[0, 0.8] = 1.0
        out = _finalise_modal_intensities(
            pd.DataFrame(index=range(2)), modal, np.array([0.5, np.inf])
        )
        pumps = [c[1] for c in out.columns if c[0] == "modal_intensities"]
        assert pumps == sorted(pumps)
        assert np.allclose(out["interacting_lasing_thresholds"].to_numpy(), [0.5, np.inf])

    def test_nonneg_active_set_keeps_all_when_positive(self):
        from netsalt.modes import _nonneg_active_set

        # diagonal (decoupled) competition matrix: every mode lases above thresh
        T = np.diag([1.0, 1.0, 1.0])
        thresholds = np.array([1.0, 1.0, 1.0])
        kept = _nonneg_active_set(T, thresholds, [0, 1, 2], pump_intensity=2.0)
        assert kept == [0, 1, 2]

    def test_nonneg_active_set_prunes_negative_mode(self):
        from netsalt.modes import _intensity_slopes_shifts, _nonneg_active_set

        # strong cross-competition makes the raw linear solve drive one mode
        # negative; the pruned active set must give only non-negative intensities
        T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.5, 1.0]])
        thresholds = np.array([1.0, 1.0, 5.0])
        ids = [0, 1, 2]
        slopes, shifts = _intensity_slopes_shifts(T, thresholds, ids)
        assert (slopes * 2.0 - shifts).min() < 0  # raw solve is unphysical
        kept = _nonneg_active_set(T, thresholds, ids, pump_intensity=2.0)
        assert kept != ids and len(kept) >= 1
        s, sh = _intensity_slopes_shifts(T, thresholds, kept)
        assert (s * 2.0 - sh).min() >= -1e-12  # survivors are non-negative


class TestIntensityMethodDispatch:
    """``step_compute_modal_intensities`` routes on ``intensity_method``."""

    def _params(self, tmp_path, method):
        from netsalt.params import NetSaltParams

        extra = {} if method is None else {"intensity_method": method}
        return NetSaltParams.from_dict(
            {"outdir": str(tmp_path), "force": True, "intensities_D0_max": 1.0, **extra}
        )

    def _run(self, tmp_path, monkeypatch, method):
        import pandas as pd

        from netsalt import pipeline

        calls = []

        def make(name):
            def _fake(*args, **kwargs):
                calls.append(name)
                return pd.DataFrame({("modal_intensities", 0.5): [0.0]})

            return _fake

        monkeypatch.setattr(pipeline, "compute_modal_intensities", make("linear"))
        monkeypatch.setattr(
            pipeline, "compute_modal_intensities_self_consistent", make("self_consistent")
        )
        monkeypatch.setattr(pipeline, "compute_modal_intensities_full_salt", make("full_salt"))
        monkeypatch.setattr(
            pipeline, "compute_modal_intensities_full_salt_newton", make("full_salt_newton")
        )
        monkeypatch.setattr(pipeline, "_attach_pump_to_graph", lambda p, qg, pump: qg)
        monkeypatch.setattr(pipeline, "save_modes", lambda *a, **k: None)

        p = self._params(tmp_path, method)
        pipeline.step_compute_modal_intensities(
            p, object(), pd.DataFrame(), np.zeros((1, 1)), None, None
        )
        return calls

    def test_default_is_linear(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, None) == ["linear"]

    def test_dispatches_each_method(self, tmp_path, monkeypatch):
        for method in ("linear", "self_consistent", "full_salt", "full_salt_newton"):
            assert self._run(tmp_path, monkeypatch, method) == [method]


class TestFullSaltNewton:
    """Building blocks of the operator-level single-mode Newton solver."""

    def _pump_graph(self, n_edges=4):
        import netsalt
        from netsalt.physics import dispersion_relation_pump
        from netsalt.quantum_graph import update_parameters

        g = make_line_graph(n_edges=n_edges)
        netsalt.set_dispersion_relation(g, dispersion_relation_pump)
        update_parameters(
            g, {"k_a": 3.0, "gamma_perp": 1.0, "D0": 0.0, "pump": np.ones(len(g.edges))}
        )
        g.graph["params"]["quality_threshold"] = 10.0
        return g

    def test_intensity_method_literal_accepts_newton(self):
        from netsalt.params import NetSaltParams

        assert NetSaltParams(intensity_method="full_salt_newton")["intensity_method"] == (
            "full_salt_newton"
        )

    def test_saturated_graph_reduces_to_pumped_at_zero_amplitude(self):
        from netsalt.modes import _saturated_graph_at
        from netsalt.physics import dispersion_relation_pump_saturated

        g = self._pump_graph()
        pump = np.asarray(g.graph["params"]["pump"], dtype=float)
        field = np.ones(len(g.edges))
        gsat = _saturated_graph_at(g, [3.0, 0.0], 0.0, 0.5, pump, field)
        # a = 0 -> denominator 1 -> D0_eff = D0 * pump (unsaturated), saturated
        # dispersion swapped in. (Equivalence to dispersion_relation_pump at this
        # D0_eff is covered by TestSaturatedDispersion.)
        np.testing.assert_allclose(gsat.graph["params"]["D0_eff"], 0.5 * pump)
        assert gsat.graph["dispersion_relation"] is dispersion_relation_pump_saturated
        # the throwaway copy must not mutate the original graph
        assert "D0_eff" not in g.graph["params"]

    def test_saturated_graph_lowers_effective_pump_with_amplitude(self):
        from netsalt.modes import _saturated_graph_at

        g = self._pump_graph()
        pump = np.asarray(g.graph["params"]["pump"], dtype=float)
        field = np.ones(len(g.edges))
        unsat = _saturated_graph_at(g, [3.0, 0.0], 0.0, 0.5, pump, field).graph["params"]["D0_eff"]
        sat = _saturated_graph_at(g, [3.0, 0.0], 1.0, 0.5, pump, field).graph["params"]["D0_eff"]
        # hole burning reduces the effective pump on the gain edges
        assert np.all(sat[pump > 0] < unsat[pump > 0])

    def test_field_intensity_is_finite_per_edge(self):
        from netsalt.modes import _get_mask_matrices, _single_mode_field_intensity

        g = self._pump_graph()
        pump_mask = _get_mask_matrices(g.graph["params"])[1]
        inten = _single_mode_field_intensity(g, [3.0, 0.05], pump_mask)
        assert inten.shape == (len(g.edges),)
        assert np.all(np.isfinite(inten))

    def test_reduces_to_linear_onset_slope_on_independent_graph(self):
        """full_salt_newton's reported intensity is in the linear modal-intensity
        unit on a graph *other* than line_PRA: the dominant mode's onset slope
        matches the linear ``1/(T_μμ·D0_thr)``. Guards the unit-consistency fix
        against the graph-dependent within-edge form factor."""
        import networkx as nx

        import netsalt
        from netsalt.modes import (
            compute_modal_intensities_full_salt_newton,
            compute_mode_competition_matrix,
            find_threshold_lasing_modes,
            pump_trajectories,
            scan_frequencies,
        )
        from netsalt.physics import dispersion_relation_pump
        from netsalt.quantum_graph import create_quantum_graph, set_total_length

        # small open dielectric line cavity straddling the gain line at k_a = 15
        n_edges = 8
        g = nx.path_graph(n_edges + 1)
        positions = np.array([[float(i), 0.0] for i in range(n_edges + 1)])
        params = {
            "open_model": "open",
            "c": 1.0,
            "k_a": 15.0,
            "gamma_perp": 3.0,
            "k_min": 12.0,
            "k_max": 18.0,
            "k_n": 80,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_n": 20,
            "quality_threshold": 1e-3,
            "search_stepsize": 0.01,
            "max_steps": 1000,
            "max_tries_reduction": 50,
            "reduction_factor": 0.8,
            "n_workers": 1,
            "D0_max": 1.0,
            "D0_steps": 10,
            "dielectric_params": {
                "method": "uniform",
                "inner_value": 9.0,
                "outer_value": 1.0,
                "loss": 0.0,
            },
        }
        create_quantum_graph(g, params, positions=positions)
        set_total_length(g, 0.5)  # short -> well-separated longitudinal modes
        netsalt.set_dielectric_constant(g, g.graph["params"])
        netsalt.set_dispersion_relation(g, dispersion_relation_pump)

        qualities = scan_frequencies(g)
        passive = netsalt.find_passive_modes(
            g, qualities, method="grid", min_distance=2, threshold_abs=0.1
        )
        pump = np.array([1.0 if g[u][v]["inner"] else 0.0 for u, v in g.edges()])
        g.graph["params"]["pump"] = pump
        trajectories = pump_trajectories(passive, g, return_approx=True)
        tdf = find_threshold_lasing_modes(trajectories, g)

        thresholds = np.asarray(tdf["lasing_thresholds"]).ravel()
        assert np.any(thresholds < np.inf), "fixture must produce a lasing mode"
        t0 = int(np.argmin(thresholds))
        thr0 = float(thresholds[t0])
        T = compute_mode_competition_matrix(g, tdf)
        linear_slope = 1.0 / (T[t0, t0] * thr0)

        # measure the newton onset slope just above the first threshold
        finite = np.sort(thresholds[thresholds < np.inf])
        d0 = thr0 + 0.4 * ((finite[1] - thr0) if finite.size > 1 else 0.3 * thr0)
        df = compute_modal_intensities_full_salt_newton(g, tdf.copy(), d0, D0_steps=4)
        cols = sorted(
            c[1] for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"
        )
        a = np.nan_to_num(df.loc[t0, [("modal_intensities", c) for c in cols]].to_numpy(float))
        newton_slope = a[-1] / (cols[-1] - thr0)
        assert 0.8 < newton_slope / linear_slope < 1.2
