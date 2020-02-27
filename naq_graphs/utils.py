"""small additiomal functions"""
import numpy as np


def _to_complex(mode):
    """convert mode array to complex number"""
    if isinstance(mode, complex):
        return mode
    return mode[0] - 1.0j * mode[1]


def _from_complex(freq):
    """convert mode array to complex number"""
    if isinstance(freq, list):
        return freq
    if isinstance(freq, np.ndarray):
        return freq
    return [np.real(freq), -np.imag(freq)]


def get_total_length(graph):
    """get the total lenght of the graph"""
    return sum([graph[u][v]["lenght"] for u, v in graph.edges()])


def get_total_inner_length(graph):
    """get the total lenght of the graph"""
    return sum(
        [graph[u][v]["lenght"] for u, v in graph.edges() if graph[u][v]["inner"]]
    )


def order_edges_by(graph, order_by_values):
    """order edges by using values in a list"""
    edges = [e for e in graph.edges]  # pylint: disable=unnecessary-comprehension
    return [edges[i] for i in np.argsort(order_by_values)]
