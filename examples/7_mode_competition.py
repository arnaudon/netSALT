import os
import sys

import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph

import naq_graphs as naq

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

os.chdir(graph_tpe)

graph, params = naq.load_graph()

if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    custom_index = []  # line PRA example
    for u, v in graph.edges:
        custom_index.append(3.0 ** 2)
    custom_index[0] = 1.0 ** 2
    custom_index[-1] = 1.0 ** 2

    count_inedges = len(graph.edges) - 2.0
    print("Number of inner edges", count_inedges)
    if count_inedges % 4 == 0:
        for i in range(round(count_inedges / 4)):
            custom_index[i + 1] = 1.5 ** 2
    else:
        print("Change number of inner edges to be multiple of 4")
    naq.set_dielectric_constant(graph, params, custom_values=custom_index)

elif graph_tpe == "line_semi":
    custom_index = []  # line OSA example
    for u, v in graph.edges:
        custom_index.append(params["dielectric_params"]["inner_value"])
    custom_index[0] = 100.0 ** 2
    custom_index[-1] = 1.0 ** 2
    naq.set_dielectric_constant(graph, params, custom_values=custom_index)

else:
    naq.set_dielectric_constant(graph, params)  # for "uniform" and all other graphs

naq.set_dispersion_relation(
    graph, naq.dispersion_relations.dispersion_relation_pump, params
)


# set pump profile for PRA example
if graph_tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
    pump_edges = round(len(graph.edges()) / 2)
    nopump_edges = len(graph.edges()) - pump_edges
    params["pump"] = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
    params["pump"][0] = 0  # first edge is outside
else:
    # params["pump"] = np.ones(len(graph.edges())) # uniform pump on ALL edges
    params["pump"] = np.zeros(len(graph.edges()))  # uniform pump on inner edges
    for i, (u, v) in enumerate(graph.edges()):
        if graph[u][v]["inner"]:
            params["pump"][i] = 1

modes, lasing_thresholds = naq.load_modes(filename="threshold_modes")
modes = np.array(modes)[np.argsort(lasing_thresholds)]
lasing_thresholds = np.array(lasing_thresholds)[np.argsort(lasing_thresholds)]
print("lasing threshold noninteracting", lasing_thresholds)

D0_max = 2*params["D0_max"]  # 10*lasing_thresholds[0] #1 #2.3
n_points = 100
pump_intensities = np.linspace(0, D0_max, n_points)
modal_intensities, interacting_lasing_thresholds = naq.compute_modal_intensities(
    graph, params, modes, lasing_thresholds, pump_intensities
)
print("Interacting thresholds:", interacting_lasing_thresholds)

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
            label="$k$" + str(i) + ": " + str(np.round(modes[i][0], decimals=2)),
        )
        #plt.axvline(lasing_thresholds[i], c=cmap.colors[n_lase % 10], ls="--")
        n_lase += 1

plt.legend()
plt.title("Uniform mode " + str(n_lase) + " lasing modes out of " + str(len(modes)))
plt.xlabel(r"$D_0$")
plt.ylabel(r"$I_\mu$")

plt.savefig("uniform_modal_pump.svg", bbox_inches="tight")

ks, alphas, qualities = pickle.load(open("scan.pkl", "rb"))
Ks = np.linspace(ks[0], ks[-1], len(ks) * 10)


def lorentzian(k, k0, gamma):
    return gamma ** 2 / ((k - k0) ** 2 + gamma ** 2)


gamma = 0.02
spectr = np.zeros(len(Ks))
I0s = []
modeks = []

plt.figure(figsize=(5, 2))
for i, intens in enumerate(modal_intensities):
    if intens[-1] > 0:
        modeks.append(modes[i][0])
        I0s.append(intens[-1])
    center = modes[i][0]
    spectr += intens[-1] * lorentzian(Ks, center, gamma)

pickle.dump([Ks, spectr], open("uniform_spectra.pkl", "wb"))

# plt.plot(Ks, spectr, '-k')
markerline, stemlines, baseline = plt.stem(modeks, I0s, '-')
markerline.set_markerfacecolor('white')
plt.setp(baseline, 'color', 'grey', 'linewidth', 1)
plt.yscale('symlog')
plt.xlabel(r"$k$")
plt.ylabel("Intensity (a.u.)")

plt.twinx()
plt.plot(Ks, lorentzian(Ks, params["k_a"], params["gamma_perp"]), "r--")

plt.ylabel("Gain spectrum (a.u.)")

plt.savefig("uniform_spectra.svg", bbox_inches="tight")
plt.show()
