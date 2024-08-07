import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from netsalt.quantum_graph import create_quantum_graph, oversample_graph
from netsalt.modes import find_modes, from_complex
from netsalt.modes import scan_frequencies
from netsalt.plotting import plot_scan, plot_single_mode
from netsalt.io import save_modes, load_modes, save_qualities, load_qualities

from netsalt.non_abelian import construct_so3_laplacian

from make_graph import make_graph

if __name__ == "__main__":
    graph, pos = make_graph(with_leads=True)
    params = {
        "open_model": "open",
        "n_workers": 7,
        "quality_threshold": 1e-7,
        "max_steps": 200,
        "k_n": 200,
        "k_min": 5.0,
        "k_max": 7.0,
        "alpha_n": 100,
        "alpha_min": -0.01,
        "alpha_max": 0.3,
        "c": len(graph.edges) * [1.0],
        "laplacian_constructor": construct_so3_laplacian,
    }
    np.random.seed(42)

    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)

    create_quantum_graph(graph, params=params, positions=pos)

    pos = [graph.nodes[u]["position"] for u in graph]
    nx.draw(graph, pos=pos)
    nx.draw_networkx_labels(graph, pos=pos)

    if not Path("qualities_uniform.h5").exists():
        qualities = scan_frequencies(graph)
        save_qualities(qualities, "qualities_uniform.h5")
    else:
        qualities = load_qualities("qualities_uniform.h5")

    if not Path("modes_uniform.h5").exists():
        modes_df = find_modes(graph, qualities)
        save_modes(modes_df, filename="modes_uniform.h5")
    else:
        modes_df = load_modes("modes_uniform.h5")
        # modes_df = load_modes("modes_open.h5")
    print(modes_df)

    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    plot_scan(graph, qualities, modes_df=modes_df, ax=ax)
    plt.savefig("so3_open_scan_uniform.pdf")

    over_graph = oversample_graph(graph, 0.01)

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 0, norm_type="real", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_1.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 0, norm_type="real_x", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_1_x.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 0, norm_type="real_y", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_1_y.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 0, norm_type="real_z", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_1_z.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 2, norm_type="real", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_2.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 2, norm_type="real_x", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_2_x.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 2, norm_type="real_y", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_2_y.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 2, norm_type="real_z", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_2_z.pdf")
    print('lkjlkj')
    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 5, norm_type="real", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_3.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 5, norm_type="real_x", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_3_x.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 5, norm_type="real_y", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_3_y.pdf")

    plt.figure(figsize=(5, 3))
    plot_single_mode(over_graph, modes_df, 5, norm_type="real_z", ax=plt.gca())
    plt.tight_layout()
    plt.savefig("so3_open_mode_3_z.pdf")
