"""collect the dispersion relations"""
import numpy as np


def dispersion_relation_linear(freq, edge_index, params=None):
    """linear dispersion relation with wavespeed"""
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq / params["c"][edge_index]


def dispersion_relation_dielectric(freq, edge_index, params=None):
    """linear dispersion relation with dielectric constant"""
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq * np.sqrt(params["dialectric_constant"][edge_index])


def dispersion_relation_pump(freq, edge_index, params=None):
    """dispersion relation with dielectric constant and pump"""
    if not params:
        raise Exception("Please provide dispersion parameters")
    if params["pump_profile"][edge_index]:
        gamma = params["gamma_perp"] / (
            np.real(freq) - params["k_a"] + 1.0j * params["gamma_perp"]
        )
        return freq * np.sqrt(
            params["dialectric_constant"][edge_index]
            + gamma * params["D0"] * params["pump_profile"][edge_index]
        )
    return freq * np.sqrt(params["dialectric_constant"][edge_index])


def set_dialectric_constant(graph, params, custom_values=None):
    """set dialectric constant in the params file using various methods"""
    if params["dialectric_params"]["method"] == "uniform":
        params["dialectric_constant"] = []
        for u, v in graph.edges:
            if graph[u][v]["inner"]:
                params["dialectric_constant"].append(
                    params["dialectric_params"]["inner_value"]
                )
            else:
                params["dialectric_constant"].append(
                    params["dialectric_params"]["outer_value"]
                )

    if params["dialectric_params"]["method"] == "random":
        params["dialectric_constant"] = np.random.normal(
            params["dialectric_params"]["mean"],
            params["dialectric_params"]["std"],
            len(graph.edges),
        )

    if params["dialectric_params"]["method"] == "custom":
        params["dialectric_constant"] = custom_values
