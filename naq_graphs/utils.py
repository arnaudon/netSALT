"""Utils functions."""
import numpy as np


def lorentzian(k, graph):
    return graph.graph["params"]["gamma_perp"] ** 2 / (
        (k - graph.graph["params"]["k_a"]) ** 2
        + graph.graph["params"]["gamma_perp"] ** 2
    )


def get_scan_grid(graph):
    """Return arrays of values to scan in complex plane."""
    ks = np.linspace(
        graph.graph["params"]["k_min"],
        graph.graph["params"]["k_max"],
        graph.graph["params"]["k_n"],
    )
    alphas = np.linspace(
        graph.graph["params"]["alpha_min"],
        graph.graph["params"]["alpha_max"],
        graph.graph["params"]["alpha_n"],
    )
    return ks, alphas


def to_complex(mode):
    """Convert mode array to complex number."""
    if isinstance(mode, complex):
        return mode
    return mode[0] - 1.0j * mode[1]


def from_complex(freq):
    """Convert mode array to complex number."""
    if isinstance(freq, list):
        return freq
    if isinstance(freq, np.ndarray):
        return freq
    return [np.real(freq), -np.imag(freq)]


def order_edges_by(graph, order_by_values):
    """Order edges by using values in a list."""
    return [list(graph.edges)[i] for i in np.argsort(order_by_values)]
