import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
from netsalt.pump import pump_cost, compute_pump_overlapping_matrix
from netsalt.io import load_graph
import numpy as np

from netsalt.plotting import plot_pump_profile


def _check(mode_id, pump, pump_overlapps):
    c0 = pump_cost(pump, mode_id, pump_overlapps)
    for i in np.array(range(len(pump)))[::-1]:
        if pump[i] == 1:
            _pump = pump.copy()
            _pump[i] = 0
            c = pump_cost(_pump, mode_id, pump_overlapps)
            if c - c0 < 1e-2:
                #print(c - c0, c)
                pump = _pump
                #c0 = c
    return _pump


if __name__ == "__main__":
    n_modes = 100
    graph = load_graph("buffon_control_LP/out/quantum_graph.gpickle")
    inner = np.array(graph.graph["params"]["inner"])

    df_uni = pd.read_hdf("../buffon_uniform/out/modal_intensities.h5")
    pump_overlapps = compute_pump_overlapping_matrix(graph, df_uni)

    base_path = Path("buffon_control_threshold/out/")
    cost_thresh = []
    for mode_id in tqdm(range(n_modes)):
        p = base_path / f"modal_intensities_{mode_id}.h5"
        with open(base_path / f"pump_profile_{mode_id}.yaml") as p_f:
            pump = np.array(yaml.safe_load(p_f))
        pump[~inner] = 0
        plt.figure()
        ax = plt.gca()
        plot_pump_profile(graph, pump, ax=ax, c="b")
        pump = _check(mode_id, pump, pump_overlapps)
        plot_pump_profile(graph, pump, ax=ax, c="r")
        c = pump_cost(pump, mode_id, pump_overlapps)
        plt.savefig(f"pumps/pump_thresh_{mode_id}.pdf")
        cost_thresh.append(c)

    base_path = Path("buffon_control_LP/out/")
    cost_LP = []
    for mode_id in tqdm(range(n_modes)):
        p = base_path / f"modal_intensities_{mode_id}.h5"
        with open(base_path / f"pump_profile_{mode_id}.yaml") as p_f:
            pump = np.array(yaml.safe_load(p_f))
        pump[~inner] = 0
        plt.figure()
        ax = plt.gca()
        plot_pump_profile(graph, pump, ax=ax, c="b")
        pump = _check(mode_id, pump, pump_overlapps)
        plot_pump_profile(graph, pump, ax=ax, c="r")
        c = pump_cost(pump, mode_id, pump_overlapps)
        plt.savefig(f"pumps/pump_LP_{mode_id}.pdf")

        c = pump_cost(pump, mode_id, pump_overlapps)
        cost_LP.append(c)

    cost_LP = np.array(cost_LP)
    cost_thresh = np.array(cost_thresh)
    plt.figure()
    plt.plot(cost_LP, "+-", label="LP")
    plt.plot(cost_thresh, "+-", label="thresh")
    plt.gca().set_ylim(0, 1.3)
    plt.legend()
    plt.savefig("cost.pdf")
