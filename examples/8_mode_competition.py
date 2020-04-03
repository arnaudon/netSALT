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

threshold_modes, lasing_thresholds = naq.load_modes(filename="threshold_modes")
# print("lasing threshold noninteracting", lasing_thresholds)

mode_competition_matrix = pickle.load(open("mode_competition_matrix.pkl", "rb"))

D0_max = 10*params["D0_max"]
n_points = 1000
pump_intensities = np.linspace(0, D0_max, n_points)
modal_intensities, interacting_lasing_thresholds = naq.compute_modal_intensities(
    np.array(threshold_modes),
    np.array(lasing_thresholds),
    pump_intensities,
    mode_competition_matrix,
)
# print("Interacting thresholds:", interacting_lasing_thresholds)

pickle.dump(
    [pump_intensities, modal_intensities, interacting_lasing_thresholds],
    open("modal_intensities_uniform.pkl", "wb"),
)

plt.figure(figsize=(5, 3))
cmap = plt.cm.get_cmap("tab10")

n_lase = 0
for i, intens in enumerate(modal_intensities):
    if intens[-1] > 0:
        plt.plot(
            pump_intensities,
            intens,
            "-",
            c=cmap.colors[n_lase % 10],
            label="$k$"
            + str(i)
            + ": "
            + str(np.round(threshold_modes[i][0], decimals=2)),
        )
        plt.axvline(lasing_thresholds[i], c=cmap.colors[n_lase % 10], ls="--")
        n_lase += 1

plt.legend()
plt.title(
    "Uniform mode " + str(n_lase) + " lasing modes out of " + str(len(threshold_modes))
)
plt.xlabel(r"$D_0$")
plt.ylabel(r"$I_\mu$")

plt.savefig("uniform_modal_pump.svg", bbox_inches="tight")

Ks = np.linspace(params["k_min"], params["k_max"], 1000)


def lorentzian(k, k0, gamma):
    return gamma ** 2 / ((k - k0) ** 2 + gamma ** 2)


gamma = 0.02
spectr = np.zeros(len(Ks))
I0s = []
modeks = []

plt.figure(figsize=(5, 2))
for i, intens in enumerate(modal_intensities):
    if intens[-1] > 0:
        modeks.append(threshold_modes[i][0])
        I0s.append(intens[-1])
    center = threshold_modes[i][0]
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
