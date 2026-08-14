"""Shared definition of the test case (netsalt side + physical constants).

Geometry: a 1D open Fabry-Perot cavity.

  * cavity: 0 < x < L,  L = 0.5, eps = n^2 = 9  (n = 3), uniformly pumped
  * outside: vacuum, purely outgoing

On the netsalt side this is a path graph whose two end edges are vacuum leads
(a vacuum lead terminated by the netsalt "open" boundary condition is a pure
transmission line, so the effective boundary condition at the cavity end is
Psi' = +- i k Psi -- exactly the continuum problem above).

The cavity is split into ``N_INNER`` equal edges. ``N_INNER`` is chosen so that
no *lasing* (real-k) mode of interest sits on the secular matrix pole
``q_e l_e in pi Z``: the modes have ``q L = pi m`` with m = 6, 7, 8, and the
pole for P equal inner edges is at ``P | m``. P = 9 avoids all three.
``noise_level=0`` keeps the geometry exactly uniform (the shipped example
jitters it, which would make an exact comparison impossible).
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np

import netsalt
from netsalt.physics import dispersion_relation_pump
from netsalt.quantum_graph import create_quantum_graph, set_total_length

L_CAV = 0.5
EPS_CAV = 9.0
N_CAV = 3.0

K_A = float(os.environ.get("NS_K_A", 15.0))
GAMMA_PERP = float(os.environ.get("NS_GAMMA_PERP", 3.0))
N_INNER = 9

PARAMS = {
    "open_model": "open",
    "c": 1.0,
    "k_a": K_A,
    "gamma_perp": GAMMA_PERP,
    "k_min": float(os.environ.get("NS_K_MIN", 11.0)),
    "k_max": float(os.environ.get("NS_K_MAX", 19.0)),
    "k_n": 400,
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_n": 100,
    "quality_threshold": 1.0e-4,
    "search_stepsize": 0.01,
    "max_steps": 10000,
    "max_tries_reduction": 50,
    "reduction_factor": 0.8,
    "n_workers": 1,
    "D0_max": float(os.environ.get("NS_D0_MAX", 1.4)),
    "D0_steps": 10,
    "dielectric_params": {
        "method": "uniform",
        "inner_value": EPS_CAV,
        "outer_value": 1.0,
        "loss": 0.0,
    },
}


def gamma_of(k, k_a=K_A, gamma_perp=GAMMA_PERP):
    """Lorentzian gain, evaluated at Re k (netsalt convention)."""
    return gamma_perp / (np.real(k) - k_a + 1.0j * gamma_perp)


def build_graph(n_inner=N_INNER, **overrides):
    n_edges = n_inner + 2
    g = nx.path_graph(n_edges + 1)
    pos = np.array([[float(i), 0.0] for i in range(n_edges + 1)])
    params = dict(PARAMS)
    params.update(overrides)
    create_quantum_graph(g, params, positions=pos, noise_level=0.0)
    set_total_length(g, L_CAV, inner=True)
    netsalt.set_dielectric_constant(g, g.graph["params"])
    netsalt.set_dispersion_relation(g, dispersion_relation_pump)
    pump_edges = os.environ.get("NS_PUMP_EDGES")
    keep = None if pump_edges is None else {int(v) for v in pump_edges.split(",")}
    pump, j = [], 0
    for u, v in g.edges():
        if not g[u][v]["inner"]:
            pump.append(0.0)
        else:
            pump.append(1.0 if (keep is None or j in keep) else 0.0)
            j += 1
    g.graph["params"]["pump"] = np.array(pump, dtype=float)
    return g


if __name__ == "__main__":
    g = build_graph()
    print("n nodes", len(g), "n edges", len(g.edges))
    print("lengths", g.graph["lengths"])
    print("inner", g.graph["params"]["inner"])
    print("pump", g.graph["params"]["pump"])
    print("inner total length", sum(g[u][v]["length"] for u, v in g.edges if g[u][v]["inner"]))
