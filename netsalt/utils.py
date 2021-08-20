"""Utils functions."""
import numpy as np

import networkx as nx


def linewidth(k, k_center, width):
    """Linewidth function."""
    return width ** 2 / ((k - k_center) ** 2 + width ** 2)


def lorentzian(k, graph):
    """Lorentzian function."""
    return linewidth(k, graph.graph["params"]["k_a"], graph.graph["params"]["gamma_perp"])


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


def _intersect(x, y):
    """Find intersection point between two segments."""

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    p = x[0]
    r = x[1] - x[0]
    q = y[0]
    s = y[1] - y[0]

    u_inter = cross(q - p, r) / cross(r, s)
    t_inter = cross(q - p, s) / cross(r, s)
    if 0 < u_inter < 1 and 0 < t_inter < 1:
        return q + u_inter * s
    return None


def _in_box(x, box):
    """Check if x is in the box."""
    if box[0] < x[0] < box[1] and box[2] < x[1] < box[3]:
        return True
    return False


def remove_pixel(graph, center, size):
    """Create the pump with missing pixel and add corresponding nodes on the graph."""
    box = [
        center[0] - 0.5 * size,
        center[0] + 0.5 * size,
        center[1] - 0.5 * size,
        center[1] + 0.5 * size,
    ]
    box_edges = np.array(
        [
            [[box[0], box[2]], [box[0], box[3]]],
            [[box[0], box[3]], [box[1], box[3]]],
            [[box[1], box[3]], [box[1], box[2]]],
            [[box[1], box[2]], [box[0], box[2]]],
        ]
    )
    ps = {}
    for box_edge in box_edges:
        for i, edge in enumerate(graph.edges):
            p = _intersect(
                box_edge, [graph.nodes[edge[0]]["position"], graph.nodes[edge[1]]["position"]]
            )
            if p is not None:
                ps[edge] = p

    for i, edge in enumerate(graph.edges):
        if _in_box(graph.nodes[edge[0]]["position"], box) and _in_box(
            graph.nodes[edge[1]]["position"], box
        ):
            graph[edge[0]][edge[1]]["pump"] = 0
        else:
            graph[edge[0]][edge[1]]["pump"] = 1

    for i, (edge, p) in enumerate(ps.items(), start=len(graph)):
        graph.add_node(i, position=p)
        graph.remove_edge(*edge)
        if _in_box(graph.nodes[edge[0]]["position"], box):
            graph.add_edge(edge[0], i, pump=0)
        else:
            graph.add_edge(edge[0], i, pump=1)
        if _in_box(graph.nodes[edge[1]]["position"], box):
            graph.add_edge(i, edge[1], pump=0)
        else:
            graph.add_edge(i, edge[1], pump=1)
    graph = nx.convert_node_labels_to_integers(graph)
    pump = [graph[e[0]][e[1]]["pump"] for e in graph.edges]
    return graph, pump
