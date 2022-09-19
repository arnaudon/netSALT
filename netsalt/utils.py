"""Some utils functions."""
import numpy as np
import networkx as nx


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


def _to_points(point, angle, t, size):
    """Convert a point/angle to set of points."""
    points = point + np.array([np.cos(angle) * t, np.sin(angle) * t]).T
    points = points[(points[:, 0] > size[0]) & (points[:, 0] < size[1])]
    return points[(points[:, 1] > size[0]) & (points[:, 1] < size[1])].tolist()


def _get_line_points(points, angles, t, size):
    """For each line, we create the points are edge list.

    We return a dict with keys are line index.
    """
    all_points = {}
    edge_list = {}
    for i, (point, angle) in enumerate(zip(points, angles)):
        _points = _to_points(point, angle, t, size)
        edge_list[i] = [(i, i + 1) for i in range(len(_points) - 1)]
        all_points[i] = _points
    return edge_list, all_points


def _get_intersection_points(points, angles, size):
    """Find the intersection points between intersecting lines.

    For each point, we return a 2-tuple with the point and indices of the intersecting lines.
    """
    intersection_points = []
    for i, (point1, angle1) in enumerate(zip(points, angles)):
        for j, (point2, angle2) in enumerate(zip(points[i:], angles[i:])):
            x = (
                point1[1] - point2[1] - np.tan(angle1) * point1[0] + np.tan(angle2) * point2[0]
            ) / (np.tan(angle2) - np.tan(angle1))
            y = point1[1] + np.tan(angle1) * (x - point1[0])
            if size[0] < x < size[1] and size[0] < y < size[1]:
                intersection_points.append([[x, y], (i, i + j)])

    return intersection_points


def _add_intersection_points(edge_list, all_points, intersection_points):
    """We add intersections point to each line by adding a new point and updating edge_list."""
    edges = []
    for intersection_point in intersection_points:
        inter_id = {}
        for i in intersection_point[1]:
            edges = edge_list[i]
            points = np.array(all_points[i])

            # search for correct segment (where intersection is in the middle)
            index = None
            for j, edge in enumerate(edges):
                x = intersection_point[0] - points[edge[0]]
                y = intersection_point[0] - points[edge[1]]
                z = points[edge[1]] - points[edge[0]]

                if abs(np.linalg.norm(x) + np.linalg.norm(y) - np.linalg.norm(z)) < 1e-10:
                    index = j

            if index is not None and inter_id is not None:
                e = edge_list[i].pop(index)
                edge_list[i].append([e[0], len(points)])
                edge_list[i].append([len(points), e[1]])
                inter_id[i] = len(points)
                all_points[i].append(intersection_point[0])
            else:
                inter_id = None

        if inter_id is not None:
            intersection_point.append(inter_id)


def _get_graph(edge_list, all_points, intersection_points):
    """We create the buffon graph by making line subgraph, and merging each intersection point.

    We return the graph and list of node positions.
    """
    graph = nx.Graph()
    shift = 0
    pos = []
    last_ids = {}
    # create the graph
    for i in edge_list:
        edges, points = edge_list[i], all_points[i]
        for edge in edges:
            graph.add_edge(edge[0] + shift, edge[1] + shift)

        last_ids[i] = shift
        shift += len(points)
        pos += points

    # merge intersection points
    for intersection_point in intersection_points:
        edge_i = intersection_point[1][0]
        edge_j = intersection_point[1][1]
        if len(intersection_point) == 3:
            i = last_ids[edge_i] + intersection_point[2][edge_i]
            j = last_ids[edge_j] + intersection_point[2][edge_j]
            graph = nx.contracted_nodes(graph, i, j)
    return graph, pos


def make_buffon_graph(n_lines, size, resolution=1.0):
    """Make a buffon graph.

    Args:
        n_lines (int): number of lines to draw randomly
        size (2-tuple): min and max extent of the graph (it will be square only)
        resolution (float): distance between each points along lines

    Warning: it is not exactly the same graph as in the Nat. Comm. Paper, which was done with
    a matlab code.
    """
    diag = np.sqrt(2) * (size[1] - size[0])
    t = np.arange(-diag, diag, resolution)
    points = np.random.uniform(size[0], size[1], size=(n_lines, 2))
    angles = np.random.uniform(0, np.pi, n_lines)

    edge_list, all_points = _get_line_points(points, angles, t, size)
    intersection_points = _get_intersection_points(points, angles, size)
    _add_intersection_points(edge_list, all_points, intersection_points)
    graph, pos = _get_graph(edge_list, all_points, intersection_points)
    return graph, pos
