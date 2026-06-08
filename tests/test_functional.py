"""Functional test of the lasing-modes pipeline."""

import os
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml
from dir_content_diff import assert_equal_trees

try:
    # dir-content-diff >= 1.x moved the pandas comparators submodule.
    from dir_content_diff.comparators import pandas as dir_content_diff_pandas
except ImportError:  # pragma: no cover - older dir-content-diff
    import dir_content_diff.pandas as dir_content_diff_pandas

import netsalt
from netsalt.config_loader import load_config
from netsalt.pipeline import compute_lasing_modes

TEST_ROOT = Path(__file__).parent
DATA = TEST_ROOT / "data"
dir_content_diff_pandas.register()


@pytest.fixture(scope="function")
def tmp_working_dir(tmp_path):
    """Change working directory before a test and change it back when the test is finished."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


def create_graph():
    # create the graph
    graph = nx.grid_2d_graph(11, 1, periodic=False)
    graph = nx.convert_node_labels_to_integers(graph)
    pos = np.array([[i / (len(graph) - 1), 0] for i in range(len(graph))])
    for n, _pos in zip(graph.nodes, pos, strict=True):
        graph.nodes[n]["position"] = _pos
    netsalt.save_graph(graph, "graph.json")

    # create the index of refraction profile
    custom_index = len(graph.edges) * [3.0**2]
    custom_loss = len(graph.edges) * [0.0]
    custom_index[0] = 1.0**2
    custom_index[-1] = 1.0**2

    count_inedges = len(graph.edges) - 2.0
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges / 4)):
            custom_index[i + 1] = 1.5**2

    with open("index.yaml", "w") as f:
        yaml.dump({"constant": custom_index, "loss": custom_loss}, f)

    # create the pump profile
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    pump = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    pump[0] = 0
    with open("pump.yaml", "w") as f:
        yaml.dump(pump.astype(int).tolist(), f)


@pytest.fixture
def working_directory(tmp_working_dir):
    """Stage the config file and the expected-output reference into the temp dir."""
    shutil.copyfile(DATA / "run_simple" / "config.yaml", tmp_working_dir / "config.yaml")
    os.mkdir(tmp_working_dir / "out")
    os.mkdir(tmp_working_dir / "figures")

    yield tmp_working_dir / "out", DATA / "run_simple" / "out"


def test_ComputeLasingModes(working_directory):
    """Run the lasing pipeline end-to-end and diff against the reference fixture."""
    create_graph()

    params = load_config("config.yaml")
    compute_lasing_modes(params)

    result_dir, expected_dir = working_directory
    assert_equal_trees(
        expected_dir, result_dir, specific_args={"out": {"patterns": [r".*\.h5$"], "atol": 1e-5}}
    )


def test_ComputeLasingModes_full_salt_newton(working_directory):
    """End-to-end smoke test of the operator-level Newton solver.

    Runs the same pipeline with ``intensity_method="full_salt_newton"`` on a
    coarse pump grid: it exercises the multimode driver, the amplitude solve and
    the frequency/profile fixed point on genuine threshold modes (no byte
    reference -- this is a coverage/regression smoke test). The dominant mode must
    lase below the max pump, matching the linear model's lowest threshold.
    """
    create_graph()

    params = load_config("config.yaml")
    params["intensity_method"] = "full_salt_newton"
    params["salt_D0_steps"] = 2
    compute_lasing_modes(params)

    result_dir, _ = working_directory
    intensities = netsalt.load_modes(str(result_dir / "modal_intensities.h5"))
    intensity_cols = [c for c in intensities.columns if c[0] == "modal_intensities"]
    assert intensity_cols, "no modal-intensity columns were written"
    # at least one mode reaches a positive intensity at the largest pump
    pumps = sorted(c[1] for c in intensity_cols)
    at_max = intensities[("modal_intensities", pumps[-1])].to_numpy(dtype=float)
    assert np.nan_to_num(at_max).max() > 0.0


def test_full_salt_newton_multimode_two_ring():
    """full_salt_newton lases *several* modes on a genuinely multimode graph.

    Two detuned rings (different sizes) joined by a bridge: the detuning localises
    each mode onto one ring, so with a narrow gain they barely compete and several
    co-lase. Guards the multimode path -- in particular against the collapse where
    a loose k-window let the solve drop a co-lasing mode by drifting its k to a
    spurious a=0 root (it then reported a single, wrong mode).
    """
    import networkx as nx

    from netsalt.modes import (
        compute_modal_intensities_full_salt_newton,
        find_passive_modes,
        find_threshold_lasing_modes,
        pump_trajectories,
    )
    from netsalt.physics import dispersion_relation_pump
    from netsalt.quantum_graph import create_quantum_graph, set_total_length

    n_a, n_b = 7, 9
    g = nx.disjoint_union(nx.cycle_graph(n_a), nx.cycle_graph(n_b))
    g.add_edge(0, n_a)
    g.add_edge(2, n_a + n_b)
    g.add_edge(n_a + 4, n_a + n_b + 1)
    pos = {}
    for i in range(n_a):
        pos[i] = [-1.6 + 0.9 * np.cos(2 * np.pi * i / n_a), 0.9 * np.sin(2 * np.pi * i / n_a)]
    for j in range(n_b):
        pos[n_a + j] = [
            1.7 + 1.25 * np.cos(2 * np.pi * j / n_b),
            1.25 * np.sin(2 * np.pi * j / n_b),
        ]
    pos[n_a + n_b] = [-1.6, -2.2]
    pos[n_a + n_b + 1] = [1.7, -2.6]
    positions = np.array([pos[i] for i in range(len(g))])
    params = {
        "open_model": "open",
        "c": 1.0,
        "k_a": 3.567,
        "gamma_perp": 0.12,
        "k_min": 3.45,
        "k_max": 3.66,
        "alpha_min": -0.05,
        "alpha_max": 0.15,
        "n_workers": 1,
        "n_modes_max": 10,
        "quality_threshold": 1e-3,
        "search_stepsize": 0.005,
        "max_steps": 1000,
        "max_tries_reduction": 50,
        "reduction_factor": 0.8,
        "D0_max": 1.0,
        "D0_steps": 14,
        "dielectric_params": {
            "method": "uniform",
            "inner_value": 9.0,
            "outer_value": 1.0,
            "loss": 0.0,
        },
    }
    create_quantum_graph(g, params, positions=positions)
    set_total_length(g, 9.0)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)

    passive = find_passive_modes(g, method="contour")
    assert len(passive) >= 2, "fixture should find several modes"
    g.graph["params"]["pump"] = np.array([1.0 if g[u][v]["inner"] else 0.0 for u, v in g.edges()])
    trajectories = pump_trajectories(passive, g, return_approx=True)
    tdf = find_threshold_lasing_modes(trajectories, g)

    df = compute_modal_intensities_full_salt_newton(g, tdf.copy(), 1.0, D0_steps=10)
    cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "modal_intensities"]
    data = np.nan_to_num(df[cols].to_numpy(dtype=float))
    peak = max(data.max(), 1e-9)
    n_lasing = int(np.sum(data.max(axis=1) > 1e-2 * peak))
    assert n_lasing >= 2, f"expected multimode lasing, got {n_lasing}"
