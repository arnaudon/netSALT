import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from netsalt.quantum_graph import oversample_graph, create_quantum_graph
from netsalt.plotting import plot_single_mode
from netsalt.physics import dispersion_relation_linear, set_dispersion_relation
from netsalt.modes import scan_frequencies, find_modes
from netsalt.plotting import plot_scan
from netsalt.io import save_modes, load_modes, save_qualities, load_qualities

from make_graph import make_graph

if __name__ == "__main__":
    graph, pos = make_graph(with_leads=True)
    params = {
        "open_model": "open",
        "n_workers": 7,
        "k_n": 200,
        "k_min": 5.0,
        "k_max": 7.0,
        "alpha_n": 100,
        "alpha_min": -0.01,
        "alpha_max": 0.3,
        "c": len(graph.edges) * [1.0],
    }
    np.random.seed(42)

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)
    plt.savefig("graph.pdf")

    create_quantum_graph(graph, params=params, positions=pos)
    set_dispersion_relation(graph, dispersion_relation_linear)

    if not Path("modes_open.h5").exists():
        qualities = scan_frequencies(graph)
        save_qualities(qualities, "qualities_open.h5")
        modes_df = find_modes(graph, qualities)
        save_modes(modes_df, filename="modes_open.h5")
    else:
        qualities = load_qualities("qualities_open.h5")
        modes_df = load_modes("modes_open.h5")
    print(modes_df)

    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    plot_scan(graph, qualities, modes_df=modes_df, ax=ax)
    plt.savefig("open_scan.pdf")

    over_graph = oversample_graph(graph, 0.01)
    over_graph.graph["params"]["c"] = len(over_graph.edges) * [1.0]

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 0, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("open_mode_1.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 2, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("open_mode_2.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 5, ax=plt.gca(), norm_type="real")
    plt.tight_layout()
    plt.savefig("open_mode_3.pdf")
