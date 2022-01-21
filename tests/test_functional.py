"""Fuctional test of the workflow."""
import shutil
import os
from pathlib import Path
import pytest
import luigi
import networkx as nx
import numpy as np
import yaml

from dir_content_diff import assert_equal_trees
import dir_content_diff.pandas

import netsalt
from netsalt.tasks.workflow import ComputeLasingModes

TEST_ROOT = Path(__file__).parent
DATA = TEST_ROOT / "data"
dir_content_diff.pandas.register()


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
    for n, _pos in zip(graph.nodes, pos):
        graph.nodes[n]["position"] = _pos
    netsalt.save_graph(graph, "graph.pkl")

    # create the index of refraction profile
    custom_index = len(graph.edges) * [3.0 ** 2]
    custom_index[0] = 1.0 ** 2
    custom_index[-1] = 1.0 ** 2

    count_inedges = len(graph.edges) - 2.0
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges / 4)):
            custom_index[i + 1] = 1.5 ** 2

    yaml.dump(custom_index, open("index.yaml", "w"))

    # create the pump profile
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    pump = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    pump[0] = 0
    yaml.dump(pump.astype(int).tolist(), open("pump.yaml", "w"))


@pytest.fixture
def working_directory(tmp_working_dir):
    """Create the working directory for the vacuum case."""
    shutil.copyfile(DATA / "run_simple" / "luigi.cfg", tmp_working_dir / "luigi.cfg")
    shutil.copyfile(DATA / "run_simple" / "logging.conf", tmp_working_dir / "logging.conf")
    os.mkdir(tmp_working_dir / "out")
    os.mkdir(tmp_working_dir / "figures")

    # Set current config in luigi
    luigi_config = luigi.configuration.get_config()
    luigi_config.read("./luigi.cfg")

    yield tmp_working_dir / "out", DATA / "run_simple" / "out"

    # Reset luigi config
    luigi_config.clear()


def test_ComputeLasingModes(working_directory):
    """Test compute lasing modes."""
    # create line graph as in Ge PRA
    create_graph()

    # run workflow
    assert luigi.build([ComputeLasingModes()], local_scheduler=True)

    # Check the numerical results
    result_dir, expected_dir = working_directory
    assert_equal_trees(expected_dir, result_dir)
