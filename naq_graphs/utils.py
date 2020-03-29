"""Utils functions."""
import numpy as np


def _to_complex(mode):
    """Convert mode array to complex number."""
    if isinstance(mode, complex):
        return mode
    return mode[0] - 1.0j * mode[1]


def _from_complex(freq):
    """Convert mode array to complex number."""
    if isinstance(freq, list):
        return freq
    if isinstance(freq, np.ndarray):
        return freq
    return [np.real(freq), -np.imag(freq)]


def order_edges_by(graph, order_by_values):
    """Order edges by using values in a list."""
    return [list(graph.edges)[i] for i in np.argsort(order_by_values)]
