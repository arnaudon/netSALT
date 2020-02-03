"""input/output functions"""
import networkx as nx
import numpy as np


def create_naq_graph(graph, params, positions=None, lengths=None):
    """append a networkx graph with necessary attributes for being a NAQ graph"""
    set_node_positions(graph, positions)

    if lengths is None:
        set_edge_lengths(graph)
    else:
        set_edge_lengths(graph, lengths=lengths)

    set_inner_edges(graph, params)


def set_inner_edges(graph, params, outer_edges=None):
    """set the inner edges to True, according to a model"""
    for u, v in graph.edges():
        if params["open_model"] == "open_ends" and (len(graph[u]) == 1 or len(graph[v]) == 1):
            graph[u][v]["inner"] = False
        elif params["open_model"] == "custom" and (u, v) in outer_edges:
            graph[u][v]["inner"] = False
        else:
            graph[u][v]["inner"] = True


def set_node_positions(graph, positions=None):
    """set the position to the networkx graph"""
    if positions is None:
        positions = nx.spring_layout(graph)
        Warning("No node positions given, plots will have random positions from spring_layout")

    for u in graph.nodes():
        graph.nodes[u]["position"] = positions[u]


def set_edge_lengths(graph, lengths=None):
    """set lengths of edges"""
    for ei, e in enumerate(list(graph.edges())):
        (u, v) = e[:2]
        if lengths is None:
            graph[u][v]["length"] = np.linalg.norm(
                graph.nodes[u]["position"] - graph.nodes[v]["position"]
            )
        else:
            graph[u][v]["length"] = lengths[ei]
