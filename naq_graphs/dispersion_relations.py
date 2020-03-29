"""Dispersion relations."""
from functools import partial

import numpy as np


def _gamma(freq, params):
    """Gamma function."""
    return params["gamma_perp"] / (
        np.real(freq) - params["k_a"] + 1.0j * params["gamma_perp"]
    )


def set_dispersion_relation(graph, dispersion_relation, params):
    """Set the dispersion relation on the graph."""
    graph.graph["dispersion_relation"] = partial(dispersion_relation, params=params)


def dispersion_relation_linear(freq, params=None):
    """Linear dispersion relation with wavespeed."""
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq / params["c"]


def dispersion_relation_dielectric(freq, params=None):
    """Linear dispersion relation with dielectric constant."""
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq * np.sqrt(params["dielectric_constant"])


def dispersion_relation_pump(freq, params=None):
    """Dispersion relation with dielectric constant and pump."""
    if not params:
        raise Exception("Please provide dispersion parameters")

    if "pump" not in params or "D0" not in params:
        return freq * np.sqrt(params["dielectric_constant"])

    return freq * np.sqrt(
        params["dielectric_constant"]
        + _gamma(freq, params) * params["D0"] * params["pump"]
    )


def set_dielectric_constant(graph, params, custom_values=None):
    """Set dielectric constant in the params file using various methods."""
    if params["dielectric_params"]["method"] == "uniform":
        for u, v in graph.edges:
            if graph[u][v]["inner"]:
                graph[u][v]["dielectric_constant"] = params["dielectric_params"][
                    "inner_value"
                ]
            else:
                graph[u][v]["dielectric_constant"] = params["dielectric_params"][
                    "outer_value"
                ]

    if params["dielectric_params"]["method"] == "random":
        for u, v in graph.edges:
            graph[u][v]["dielectric_constant"] = np.random.normal(
                params["dielectric_params"]["mean"],
                params["dielectric_params"]["std"],
                1,
            )

    if params["dielectric_params"]["method"] == "custom":
        for ei, e in enumerate(graph.edges):
            graph[e[0]][e[1]]["dielectric_constant"] = custom_values[ei]

    update_params_dielectric_constant(graph, params)


def update_params_dielectric_constant(graph, params):
    """Update the dielectric constant values in the params dictionary."""
    params["dielectric_constant"] = [
        graph[u][v]["dielectric_constant"] for u, v in graph.edges
    ]
