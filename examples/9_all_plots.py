import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import netsalt
from netsalt import plotting

if __name__ == "__main__":
    if len(sys.argv) > 1:
        graph_tpe = sys.argv[-1]
    else:
        print("give me a type of graph please!")

    params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

    os.chdir(graph_tpe)

    graph = netsalt.load_graph()
    graph = netsalt.oversample_graph(graph, params)

    modes_df = netsalt.load_modes()
    qualities = netsalt.load_qualities()

    linewidth = 1 / (params["innerL"] * params["k_a"])
    if linewidth < 5e-4:
        linewidth = 5.0e-4
    plotting.plot_spectra(graph, modes_df, width=linewidth)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    plotting.plot_ll_curve(
        graph, modes_df, with_legend=False, with_colors=True, with_thresholds=False, ax=ax
    )

    ll_axis = inset_axes(ax, width="40%", height="40%", borderpad=2, loc="upper left")
    plotting.plot_ll_curve(
        graph,
        modes_df,
        with_legend=False,
        with_colors=True,
        with_thresholds=False,
        ax=ll_axis,
    )

    D0s = modes_df["modal_intensities"].columns.values
    top = np.max(np.nan_to_num(modes_df["modal_intensities"].to_numpy()[0, round(0.3 * len(D0s))]))
    ll_axis.axis([D0s[0], D0s[round(0.3 * len(D0s))], -0.01, top])

    ll_axis.tick_params(axis="both", which="major", labelsize=8)
    ll_axis.xaxis.label.set_size(9)
    ll_axis.set_ylabel("")

    fig.savefig("ll_curves.png", bbox_inches="tight")

    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(wspace=0, hspace=0)
    plotting.plot_stem_spectra(graph, modes_df, -1, ax=axes[0])
    axes[0].set_xticks([])
    plotting.plot_scan(graph, qualities, modes_df, ax=axes[1])
    plotting.plot_pump_traj(modes_df, with_scatter=False, with_approx=False, ax=axes[1])

    fig.savefig("final_plot.png", bbox_inches="tight")
    plt.show()

    lasing_mode_id = np.argsort(modes_df["interacting_lasing_thresholds"].to_numpy())

    # plot first few modes in order of interacting lasing threshold
    fig, axes = plt.subplots(
        # nrows=int(np.ceil(len(lasing_mode_id) / 3.0)), ncols=3, figsize=(12, 4)
        nrows=3,
        ncols=3,
        figsize=(12, 12),
    )
    for ax, index in zip(axes.flatten(), lasing_mode_id):
        plotting.plot_single_mode(
            graph, modes_df, index, df_entry="threshold_lasing_modes", colorbar=False, ax=ax
        )

    fig.savefig("lasing_modes_orderedby_Dth.png", bbox_inches="tight")

    # lasing modes in order of intensity at specific pump
    lasing_modes_list = np.where(
        np.nan_to_num(modes_df["modal_intensities"].to_numpy()[:, -1]) > 0
    )[0]
    lasing_modes_ordered = np.argsort(
        np.nan_to_num(modes_df["modal_intensities"].to_numpy()[:, -1])
    )[::-1]
    lasing_mode_id = lasing_modes_ordered[range(len(lasing_modes_list))]
    # pickle.dump(lasing_mode_id, open("modelist.pkl", "wb"))
    print("lasing modes: ", lasing_mode_id)

    # plot modes corresponding to largest peaks in spectrum
    fig, axes = plt.subplots(
        # nrows=int(np.ceil(len(lasing_mode_id) / 3.0)), ncols=3, figsize=(12, 4)
        nrows=3,
        ncols=3,
        figsize=(12, 12),
    )
    for ax, index in zip(axes.flatten(), lasing_mode_id):
        plotting.plot_single_mode(
            graph, modes_df, index, df_entry="threshold_lasing_modes", colorbar=False, ax=ax
        )

    fig.savefig("lasing_modes.png", bbox_inches="tight")

    plt.show()
