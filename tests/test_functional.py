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
