from netsalt.plotting import plot_spectra, get_spectra, plot_pump_traj
import pandas as pd
import numpy as np
from netsalt.io import load_graph
from netsalt.io import load_modes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    graph = load_graph('out/quantum_graph.gpickle')
    graph.graph["params"]['k_min'] = 10.35
    graph.graph["params"]['k_max'] = 11.
    graph.graph["params"]['k_n'] = 1500
    graph.graph["params"]['alpha_min'] = 10.35
    graph.graph["params"]['alpha_max'] = 11.
    graph.graph["params"]['alpha_n'] = 1500

    missing_modes = 'out/modal_intensities.h5'
    missing_modes_df = load_modes(missing_modes)

    missing_threshold_modes = np.real(missing_modes_df["threshold_lasing_modes"])
    missing_modal_amplitudes = np.real(missing_modes_df["modal_intensities"].iloc[:, -1])

    modes = '../buffon_example/out/modal_intensities_5.h5'
    modes_df = load_modes(modes)

    ### plot mode traj
    plt.figure(figsize=(5, 2))
    ax = plt.gca()
    pumped_modes = modes_df["mode_trajectories"].to_numpy()
    _m = modes_df[abs(np.real(modes_df['passive']) -10.629930)<0.00001]
    print(_m)
    #ax.plot(np.real(_m['passive']), np.imag(_m['passive']), 'o', c='r' )
    for pumped_mode in pumped_modes:
        ax.plot(np.real(pumped_mode)[0], np.imag(pumped_mode)[0], 'o', c='k' )
        ax.plot(np.real(pumped_mode), np.imag(pumped_mode),'-', c='k', lw=1.0)

    pumped_modes = missing_modes_df["mode_trajectories"].to_numpy()
    _m = modes_df[abs(np.real(modes_df['passive']) -10.629930)<0.00001]
    print(_m)
    for pumped_mode in pumped_modes:
        ax.plot(np.real(pumped_mode), np.imag(pumped_mode), '--', lw=1.0, c='k' )

    plt.axhline(0, c='k')
    ax.set_xlim(10.627 , 10.633)
    plt.xlabel('Real(k)')
    plt.ylabel('Im(k)')
    ax.set_ylim(np.min(np.imag(pumped_mode)), 0.001)
    plt.savefig('mode_traj.pdf', bbox_inches='tight')


    ### plot spectra diff
    ks, spectra_missing = get_spectra(graph, missing_modes_df)
    ks, spectra = get_spectra(graph, modes_df)

    plt.figure(figsize=(5, 3))
    plt.plot(ks, spectra_missing, lw=0.8)
    plt.plot(ks, spectra, lw=0.8)

    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mode amplitude (a.u)")
    plt.savefig('all.pdf')

    plt.figure(figsize=(5, 2))
    plt.plot(ks, spectra - spectra_missing, lw=0.8)
    plt.savefig('diff.pdf')
