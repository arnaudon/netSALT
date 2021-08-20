import math

import networkx as nx
import numpy as np
from networkx.generators.harary_graph import hkn_harary_graph, hnm_harary_graph

import netsalt


def generate_pump(tpe, graph, params):
    if tpe == "line_PRA":
        # set pump profile for PRA example
        pump_edges = round(len(graph.edges()) / 2)
        nopump_edges = len(graph.edges()) - pump_edges
        params["pump"] = np.append(np.ones(pump_edges), np.zeros(nopump_edges))
        params["pump"][0] = 0  # first edge is outside

    elif tpe == "buffon":
        # uniform pump on inner edges
        params["pump"] = np.zeros(len(graph.edges()))
        for i, (u, v) in enumerate(graph.edges()):
            if graph[u][v]["inner"]:
                params["pump"][i] = 1
        if params["pump_edges"] != "all":
            if params["pump_edges"] == "centre":
                # switch off pump on edges based on node position
                positions = [graph.nodes[u]["position"] for u in graph]
                for i, (u, v) in enumerate(graph.edges()):
                    if (
                        np.linalg.norm(positions[u]) < 25
                        and np.linalg.norm(positions[v]) < 25
                    ):
                        params["pump"][i] = 0
            elif params["pump_edges"] == "mode":
                # pump pattern using mode profile
                modes_df = netsalt.load_modes()
                mode = modes_df["passive"][19]
                edge_solution = mean_mode_on_edges(mode, graph)
                params["pump"] = np.where(
                    edge_solution > 0.1 * max(edge_solution), 0, 1
                )
                for i, (u, v) in enumerate(graph.edges()):
                    if graph[u][v]["inner"] == False:
                        params["pump"][i] = 0
            else:
                # switch off pump on given set of edges
                off_edges = np.array(params["pump_edges"])
                for i, j in enumerate(off_edges):
                    params["pump"][j] = 0

    elif tpe == "knot":
        # uniform pump on inner edges
        params["pump"] = np.zeros(len(graph.edges()))
        for i, (u, v) in enumerate(graph.edges()):
            if graph[u][v]["inner"]:
                params["pump"][i] = 1
        if params["pump_edges"] != "all":
            # switch off pump on given set of edges
            off_edges = np.array(params["pump_edges"])
            for i, j in enumerate(off_edges):
                params["pump"][j] = 0

    else:
        # uniform pump on inner edges
        params["pump"] = np.zeros(len(graph.edges()))
        for i, (u, v) in enumerate(graph.edges()):
            if graph[u][v]["inner"]:
                params["pump"][i] = 1

        print("Using uniform pump.")

    graph.graph["params"]["pump"] = params["pump"]


def generate_index(tpe, graph, params):
    """Set non-uniform dielectric constants."""
    if tpe == "line_PRA" and params["dielectric_params"]["method"] == "custom":
        custom_index = []  # line PRA example
        for u, v in graph.edges:
            custom_index.append(3.0 ** 2)
        custom_index[0] = 1.0 ** 2
        custom_index[-1] = 1.0 ** 2

        count_inedges = len(graph.edges) - 2.0
        print("Number of inner edges", count_inedges)
        if count_inedges % 4 == 0:
            for i in range(round(count_inedges / 4)):
                custom_index[i + 1] = 1.5 ** 2
        else:
            print("Change number of inner edges to be multiple of 4")
        return custom_index

    if tpe == "line_semi":
        custom_index = []  # line OSA example or Esterhazy PRA 2014
        for u, v in graph.edges:
            custom_index.append(params["dielectric_params"]["inner_value"])
        custom_index[0] = 100.0 ** 2
        custom_index[-1] = 1.0 ** 2
        return custom_index

    if tpe == "coupled_rings" or tpe == "knot":
        custom_index = []
        for u, v in graph.edges:
            if v <= params["n"]:
                custom_index.append(
                    (
                        params["refraction_params"]["inner_value"]
                        + 1.0j * params["refraction_params"]["loss"]
                    )
                    ** 2
                )
            else:
                custom_index.append(
                    (
                        params["refraction_params"]["inner_value"]
                        * params["refraction_params"]["detuning"]
                        + 1.5j * params["refraction_params"]["loss"]
                    )
                    ** 2
                )  # change index on second cavity

        custom_index[2] = (
            custom_index[2] * params["refraction_params"]["coupling"]
        )  # change index on linking edge
        # custom_index[2] = (
        #    params["refraction_params"]["inner_value"]
        #    + 1.0j * params["refraction_params"]["coupling"]
        # )**2  # change loss on linking edge
        if tpe == "knot":
            custom_index[params["n"] + 1] = (
                params["refraction_params"]["outer_value"] ** 2
            )
            custom_index[-1] = params["refraction_params"]["outer_value"] ** 2

        return custom_index
    return None


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
        G = nx.connected_watts_strogatz_graph(params['n'], params['k'], params['p'], seed = params['seed'])
        #G = nx.watts_strogatz_graph(params['n'], params['k'], params['p'], seed = params['seed'])
        pos = [
            np.array([np.cos(2 * np.pi * i / len(G)), np.sin(2 * np.pi * i / len(G))])
            for i in range(len(G))
        ]

    if tpe == "hnm_harary":
        G = hnm_harary_graph(
            params["n"], params["m"]
        )
        pos = np.array(list(nx.spring_layout(G).values()))
        #pos = np.array(list(nx.spectral_layout(G).values()))
        #pos = np.array(list(nx.kamada_kawai_layout(G).values()))

    if tpe == "hkn_harary":
        G = hkn_harary_graph(
            params["k"], params["n"]
        )
        pos = np.array(list(nx.spring_layout(G).values()))
        #pos = np.array(list(nx.spectral_layout(G).values()))
        #pos = np.array(list(nx.kamada_kawai_layout(G).values()))

    elif tpe == "ER":
        G = nx.erdos_renyi_graph(params["n"], params["p"])

    elif tpe == "grid":
        G = nx.grid_2d_graph(params["n"]+1, params["m"]+1, periodic=False)
        pos = []
        for u in G.nodes:
            pos.append(np.array(u, dtype=np.float))
        G = nx.convert_node_labels_to_integers(G)
        #re-centre graph to origin
        pos = np.array(pos)
        offset = (pos.max(axis=0) - pos.min(axis=0))/2
        #print(offset)
        pos -= offset

    elif tpe == "hexgrid":
        G = nx.hexagonal_lattice_graph(params["n"], params["m"], periodic=False, with_positions=True)
        pos_dic = nx.get_node_attributes(G, 'pos')
        pos = []
        for u in G.nodes:
            pos.append(np.array(pos_dic[u], dtype=np.float))
        G = nx.convert_node_labels_to_integers(G)
        #re-centre graph to origin
        pos = np.array(pos)
        offset = (pos.max(axis=0) - pos.min(axis=0))/2
        #print(offset)
        pos -= offset

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
        pos += [np.array([-ringL / 2 - 0.5, -1.000015])]
        pos += [
            np.array([ringL * (i / (nline - 1) - 0.5), -1.000015]) for i in range(nline)
        ]
        pos += [np.array([ringL / 2 + 0.5, -1.000015])]
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
        #pos = np.array(list(nx.spring_layout(G).values()))
        #pos = np.array(list(nx.spectral_layout(G).values()))
        pos = np.array(list(nx.kamada_kawai_layout(G).values()))

    elif tpe == "rtree":
        G = nx.full_rary_tree(params["r"], params["h"])
        G = nx.convert_node_labels_to_integers(G)
        pos = np.array(list(nx.spring_layout(G).values()))
        #pos = np.array(list(nx.spectral_layout(G).values()))
        #pos = np.array(list(nx.kamada_kawai_layout(G).values()))

    elif tpe == "buffon":
        import scipy.io as io

        mat = io.loadmat("datasets/buffon_likeExp.mat")
        GG = nx.Graph(mat["Adj"])
        # get only the largest connected component
        GG = GG.subgraph(max(nx.connected_components(GG), key=len))
        G = nx.Graph(GG)
        pos = mat["V"][list(G.nodes)]
        #pos = np.array(list(nx.spring_layout(G).values()))
        #pos = np.array(list(nx.spectral_layout(G).values()))
        #pos = np.array(list(nx.kamada_kawai_layout(G).values()))

    elif tpe == "buffon_grid":
        import scipy.io as io

        mat = io.loadmat("datasets/buffon_graph_grid.mat")
        GG = nx.Graph(mat["Adj"])
        # get only the largest connected component
        GG = GG.subgraph(max(nx.connected_components(GG), key=len))
        G = nx.Graph(GG)
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
            pos = np.append(pos, [pos[n] * 1.04], axis=0)
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
