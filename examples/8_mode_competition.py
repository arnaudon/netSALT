import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

import naq_graphs as naq
from graph_generator import generate_graph

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = naq.load_graph()
naq.update_parameters(graph, params)

modes_df = naq.load_modes()

mode_competition_matrix = pickle.load(open("mode_competition_matrix.pkl", "rb"))

D0_max = params["D0_max"]
n_points = 1000
pump_intensities = np.linspace(0, D0_max, n_points)
modes_df = naq.compute_modal_intensities(
    modes_df, pump_intensities, mode_competition_matrix
)

naq.save_modes(modes_df)

plt.figure(figsize=(5, 3))
cmap = plt.cm.get_cmap("tab10")

n_lase = 0
for i, intens in enumerate(modes_df["modal_intensities"].to_numpy()):
    if intens[-1] > 0:
        plt.plot(
            pump_intensities,
            intens,
            "-",
            c=cmap.colors[n_lase % 10],
            label="$k$"
            + str(i)
            + ": "
            + str(
                np.round(np.real(modes_df.loc[i, "threshold_lasing_modes"]), decimals=2)
            ),
        )
        plt.axvline(
            modes_df.loc[i, "lasing_thresholds"], c=cmap.colors[n_lase % 10], ls="--"
        )
        n_lase += 1

plt.legend()
plt.title("Uniform mode " + str(n_lase) + " lasing modes out of " + str(len(modes_df)))
plt.xlabel(r"$D_0$")
plt.ylabel(r"$I_\mu$")

plt.savefig("uniform_modal_pump.svg", bbox_inches="tight")

Ks = np.linspace(graph.graph["params"]["k_min"], graph.graph["params"]["k_max"], 1000)


def lorentzian(k, k0, gamma):
    return gamma ** 2 / ((k - k0) ** 2 + gamma ** 2)


gamma = 0.02
spectr = np.zeros(len(Ks))
I0s = []
modeks = []

plt.figure(figsize=(5, 2))
for i, intens in enumerate(modes_df["modal_intensities"].to_numpy()):
    if intens[-1] > 0:
        modeks.append(np.real(modes_df.loc[i, "threshold_lasing_modes"]))
        I0s.append(intens[-1])
    center = modeks[-1]
    spectr += intens[-1] * lorentzian(Ks, center, gamma)

pickle.dump([Ks, spectr], open("uniform_spectra.pkl", "wb"))

# plt.plot(Ks, spectr, '-k')
markerline, stemlines, baseline = plt.stem(modeks, I0s, "-")
markerline.set_markerfacecolor("white")
plt.setp(baseline, "color", "grey", "linewidth", 1)
plt.yscale("symlog")
plt.xlabel(r"$k$")
plt.ylabel("Intensity (a.u.)")

plt.twinx()
plt.plot(Ks, lorentzian(Ks, params["k_a"], params["gamma_perp"]), "r--")

plt.ylabel("Gain spectrum (a.u.)")

plt.savefig("uniform_spectra.svg", bbox_inches="tight")
plt.show()
