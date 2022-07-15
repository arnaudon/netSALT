"""Some utils functions."""
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.spatial import distance_matrix


def linewidth(k, k_center, width):
    """Linewidth function.

    Args:
        k (float): wavenumber
        k_center (float): wavenumber for center of linewidth
        width (float): width of linewidth
    """
    return width**2 / ((k - k_center) ** 2 + width**2)


def lorentzian(k, graph):
    """Lorentzian function using linewidth.

    Args:
        k (float): wavenumber
        graph (graph): graph with linewidth parameters k_a and gamma_perp
    """

    return linewidth(k, graph.graph["params"]["k_a"], graph.graph["params"]["gamma_perp"])


def get_scan_grid(graph):
    """Return arrays of values to scan in complex plane from graph parameters.

    graph (graph): graph with wavenumber scan parameters
    """
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


def make_buffon_graph(n_points, size, resolution=1.0):
    diag = np.sqrt(2) * (size[1] - size[0])
    t = np.linspace(-diag, diag, int(2.0 * (size[1] - size[0]) / resolution))

    def _to_points(point, angle):
        points = point + np.array([np.cos(angle) * t, np.sin(angle) * t]).T
        points = points[(points[:, 0] > size[0]) & (points[:, 0] < size[1])]
        return points[(points[:, 1] > size[0]) & (points[:, 1] < size[1])].tolist()

    points = np.random.uniform(size[0], size[1], size=(n_points, 2))
    angles = np.random.uniform(0, np.pi, n_points)

    def get_line_points():
        all_points = []
        edge_list = []
        current_n_points = 0
        for point, angle in zip(points, angles):
            _points = _to_points(point, angle)
            edge_list += [
                (current_n_points + i, current_n_points + i + 1) for i in range(len(_points) - 1)
            ]
            current_n_points += len(_points)
            all_points += _points
        return edge_list, np.array(all_points)

    def get_adjacency(all_points, edge_list):
        adjacency = np.zeros(2 * [len(all_points)])
        for edge in edge_list:
            adjacency[edge] = 1.0
        adjacency += adjacency.T
        return adjacency

    def get_new_nodes(all_points):
        dist = distance_matrix(all_points, all_points)
        mask = dist <= resolution
        dist[mask] = 1.0
        dist[~mask] = 0.0
        dist -= np.diag(np.diag(dist))

        new_nodes_ids = []
        for i in range(len(dist)):
            if len(dist[dist[i] == 1]) > 0:
                _ids = np.argwhere(dist[i] == 1).flatten().tolist() + [i]
                new_nodes_ids.append(set(_ids))
        return new_nodes_ids

    def clean_new_modes(new_nodes_ids):
        cleaned_nodes_ids = []
        for _ids in new_nodes_ids:
            add = True
            for _ids2 in new_nodes_ids:
                if _ids.issubset(_ids2):
                    if _ids2 != _ids:
                        add = False
            if add and _ids not in cleaned_nodes_ids:
                cleaned_nodes_ids.append(list(_ids))

        _new_nodes = []
        for _ids in cleaned_nodes_ids:
            _new_nodes.append(np.mean(all_points[list(_ids)], axis=0))

        new_dist = distance_matrix(_new_nodes, _new_nodes)
        mask = new_dist < 2 * resolution
        new_dist[mask] = 1
        new_dist[~mask] = 0

        components = connected_components(new_dist)[1]
        collapsed_nodes_ids = []
        for comp in np.unique(components):
            ids = np.argwhere(components == comp).flatten()
            _n = []
            for _id in ids:
                _n += cleaned_nodes_ids[_id]
            collapsed_nodes_ids.append(set(_n))

        return [list(n) for n in collapsed_nodes_ids]

    def make_adjacency(cleaned_nodes_ids, all_points):
        new_nodes = []
        for _ids in cleaned_nodes_ids:
            new_nodes.append(np.mean(all_points[list(_ids)], axis=0))

        all_points = np.array(all_points.tolist() + new_nodes)

        new_adjacency = np.zeros(2 * [len(adjacency) + len(new_nodes)])
        new_adjacency[: len(adjacency)][:, : len(adjacency)] = adjacency
        import itertools

        cross_mask = np.argwhere(new_adjacency.sum(1) == 1).flatten()
        for new_id, _ids in enumerate(cleaned_nodes_ids):
            for i, j in itertools.combinations(_ids, 2):
                new_adjacency[i, :] = 0
                new_adjacency[:, i] = 0
                new_adjacency[j, :] = 0
                new_adjacency[:, j] = 0
            nodes = [
                n for n in np.argwhere(new_adjacency.sum(1) == 1).flatten() if n not in cross_mask
            ]
            for node in nodes:
                new_node = new_id + len(adjacency)
                new_adjacency[node, new_node] = 1
                new_adjacency[new_node, node] = 1
        return new_adjacency, all_points

    print(1)
    edge_list, all_points = get_line_points()
    print(2)
    adjacency = get_adjacency(all_points, edge_list)
    print(3)
    new_nodes_ids = get_new_nodes(all_points)
    print(4)
    cleaned_nodes_ids = clean_new_modes(new_nodes_ids)
    print(5)
    new_adjacency, all_points = make_adjacency(cleaned_nodes_ids, all_points)
    import networkx

    graph = nx.Graph(new_adjacency)
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.scatter(*np.array(new_nodes).T, s=4, c="g")
    nx.draw(graph, pos=all_points, node_size=0.00, width=0.2)
    plt.savefig("test.pdf")
