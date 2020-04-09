import networkx as nx
import numpy as np
import math


def generate_graph(tpe="SM", params={}):

    pos = []
    np.random.seed(params["seed"])

    if tpe == "SM_small":
        G = nx.newman_watts_strogatz_graph(
            params["n"], params["k"], params["p"], seed=params["seed"]
        )

        pos = np.array(
            [
                [np.cos(2 * np.pi * i / len(G)), np.sin(2 * np.pi * i / len(G))]
                for i in range(len(G))
            ]
        )

    elif tpe == "SM_0" or tpe == "SM_1" or tpe == "SM_2" or tpe == "SM_0_1":
        # G = nx.connected_watts_strogatz_graph(params['n'], params['k'], params['p'], seed = params['seed'])
        G = nx.newman1watts_strogatz_graph(10, 2, 0)  # , seed = params['seed'])
        pos = [
            np.array([np.cos(2 * np.pi * i / len(G)), np.sin(2 * np.pi * i / len(G))])
            for i in range(len(G))
        ]

    elif tpe == "ER":
        G = nx.erdos_renyi_graph(params["n"], params["p"])

    elif tpe == "grid":
        G = nx.grid_2d_graph(params["n"], params["m"], periodic=True)
        G = nx.convert_node_labels_to_integers(G)

    elif tpe == "line" or tpe == "line_semi" or tpe == "line_PRA":
        G = nx.grid_2d_graph(params["n"], 1, periodic=False)
        G = nx.convert_node_labels_to_integers(G)
        pos = np.array([[i / (len(G) - 1), 0] for i in range(len(G))])

    elif tpe == "coupled_rings":
        edges = []
        for i in range(params["n"] - 1):
            edges.append((i, i + 1))
        edges.append((0, params["n"] - 1))
        for i in range(params["n"] - 1):
            edges.append((params["n"] + i, params["n"] + i + 1))
        edges.append((params["n"], 2 * params["n"] - 1))
        edges.append((0, params["n"]))

        G = nx.Graph(edges)
        pos = [
            np.array(
                [
                    np.cos(2 * np.pi * i / params["n"]),
                    np.sin(2 * np.pi * i / params["n"]),
                ]
            )
            for i in range(params["n"])
        ]
        pos += [
            np.array(
                [
                    2.1 - np.cos(2 * np.pi * i / params["n"]),
                    np.sin(2 * np.pi * i / params["n"]),
                ]
            )
            for i in range(params["n"])
        ]
        pos = np.array(pos)

    elif tpe == "knot":
        edges = []
        for i in range(params["n"] - 1):
            edges.append((i, i + 1))
        edges.append((0, params["n"] - 1))

        for i in range(params["n"] + 1):
            edges.append((params["n"] + i, params["n"] + i + 1))

        edges.append((0, math.ceil(len(edges) - params["n"] / 2) - 1))

        G = nx.Graph(edges)
        pos = [
            np.array(
                [
                    np.sin(2 * np.pi * i / params["n"]),
                    -np.cos(2 * np.pi * i / params["n"]),
                ]
            )
            for i in range(params["n"])
        ]
        ringL = 2 * params["n"] * np.sin(np.pi / params["n"])

        nline = round(len(G) / 2) - 1
        pos += [np.array([-ringL / 2 - 0.5, -1.05])]
        pos += [
            np.array([ringL * (i / (nline - 1) - 0.5), -1.05]) for i in range(nline)
        ]
        pos += [np.array([ringL / 2 + 0.5, -1.05])]
        pos = np.array(pos)

    elif tpe == "SBM" or tpe == "SBM_2":
        import SBM as sbm

        G = nx.stochastic_block_model(
            params["sizes"],
            np.array(params["probs"]) / params["sizes"][0],
            seed=params["seed"],
        )
        for i in G:
            G.node[i]["old_label"] = G.node[i]["block"]

        # G,community_labels= sbm.SBM_graph(params['n'], params['n_comm'], params['p'])

    elif tpe == "powerlaw":
        G = nx.powerlaw_cluster_graph(params["n"], params["m"], params["p"])

    elif tpe == "geometric":
        G = nx.random_geometric_graph(params["n"], params["p"])

    elif tpe == "tree":
        G = nx.balanced_tree(params["r"], params["h"])
        G = nx.convert_node_labels_to_integers(G)
        pos = np.array(list(nx.spring_layout(G).values()))

    elif tpe == "buffon":
        import scipy.io as io

        mat = io.loadmat("datasets/buffonX_log3.mat")
        G = nx.Graph(mat["Adj"])
        # get only the largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len))
        pos = mat["V"][list(G.nodes)]

    elif (
        tpe == "uniform_delaunay_0"
        or tpe == "uniform_delaunay_1"
        or tpe == "uniform_delaunay_2"
    ):

        from scipy.spatial import Delaunay

        # np.random.seed(0)

        points = []
        x = np.linspace(-2.0, 2.0, params["n"])
        points = []
        for i in range(params["n"]):
            for j in range(params["n"]):
                point_test = np.array(
                    [
                        x[j] + np.random.normal(0, params["eps"]),
                        x[i] + np.random.normal(0, params["eps"]),
                    ]
                )
                if np.linalg.norm(point_test) < 1.1:
                    points.append(point_test)

        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0], t[1]])
            edge_list.append([t[0], t[2]])
            edge_list.append([t[1], t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()

        pos = list(pos)

    elif tpe == "delaunay-bump":

        from scipy.spatial import Delaunay

        # np.random.seed(0)
        x = np.linspace(0, 1, params["n"])

        points = []
        for i in range(params["n"]):
            point_test = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(point_test) < 1:
                points.append(point_test)
        #    for j in range(params['n']):
        #        points.append([x[j]+np.random.normal(0,params['eps']),x[i]+np.random.normal(0,params['eps'])])

        for i in range(params["n_sub"]):
            point_test = np.random.normal(0.3, 0.15, 2)
            points.append(point_test)

        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0], t[1]])
            edge_list.append([t[0], t[2]])
            edge_list.append([t[1], t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()

        for i, j in G.edges:
            G[i][j]["weight"] = 1.0 / np.linalg.norm(points[i] - points[j])

        pos = list(pos)

    elif tpe == "delaunay-disk":

        from scipy.spatial import Delaunay

        # np.random.seed(0)
        x = np.linspace(0, 1, params["n"])

        points = []
        for i in range(params["n"]):
            point_test = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(point_test) < 1:
                points.append(point_test)
        #    for j in range(params['n']):
        #        points.append([x[j]+np.random.normal(0,params['eps']),x[i]+np.random.normal(0,params['eps'])])

        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0], t[1]])
            edge_list.append([t[0], t[2]])
            edge_list.append([t[1], t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()

        for i, j in G.edges:
            G[i][j]["weight"] = 1.0 / np.linalg.norm(points[i] - points[j])

        pos = list(pos)

    if params["lead_prob"] > 0:
        # add infinite leads to the network with probability lead_prob
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pos)
        k = 0
        n_tot = len(G)
        hull_nodes = set(hull.simplices.flatten())
        print(len(hull_nodes), "hull nodes")

        for n in hull_nodes:
            p = np.random.rand()
            if p < params["lead_prob"]:
                G.add_node(n_tot + k)
                G.add_edge(n, n_tot + k)
                pos = np.append(pos, [pos[n] * 1.4], axis=0)
                k += 1

    elif params["lead_prob"] == -1:

        # add infinite leads to all nodes in the network
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pos)
        k = 0
        n_tot = len(G)
        hull_nodes = set(hull.simplices.flatten())
        print(len(hull_nodes), "hull nodes")

        for n in hull_nodes:
            G.add_node(n_tot + k)
            G.add_edge(n, n_tot + k)
            pos = np.append(pos, [pos[n] * 1.4], axis=0)
            k += 1

    elif params["lead_prob"] == -2:

        # add infinite leads to the network on specific nodes
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pos)
        k = 0
        n_tot = len(G)
        hull_nodes = set(hull.simplices.flatten())
        print(len(hull_nodes), "hull nodes")

        # for n in hull_nodes[-5:]:
        for n in hull_nodes[:1]:
            G.add_node(n_tot + k)
            G.add_edge(n, n_tot + k)
            pos = np.append(pos, [pos[n] * 2.73], axis=0)  # pos[n]*1.4
            k += 1

    return G, pos
