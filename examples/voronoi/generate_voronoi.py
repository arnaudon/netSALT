import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d

from netsalt.io import save_graph

def generate_hex_grid(radius=1.0, grid_size=3):
    """Generates coordinates for a hexagonal grid.
    
    # Parameters
    radius = 1.0   # Distance from the center to a vertex of the hexagon
    grid_size = 3  # Defines the range of the grid in axial coordinates
    """
    
    hex_points = []
    
    # The vertical distance between adjacent rows (based on hexagon geometry)
    row_height = np.sqrt(3) * radius
    
    # Generate the grid of points
    for q in range(-grid_size, grid_size + 1):
        r_min = max(-grid_size, -q - grid_size)
        r_max = min(grid_size, -q + grid_size)
        for r in range(r_min, r_max + 1):
            # Convert axial coordinates (q, r) into 2D cartesian coordinates (x, y)
            x = radius * 3/2 * q
            y = row_height * (r + q / 2)
            hex_points.append((x, y))
    
    return hex_points

def generate_square_grid(N=5):
    """Generates coordinates for a square grid.
    
    # Define the grid size (N x N)
    eg. N = 5
    """

    # Generate x and y coordinates
    x = np.arange(N) - (N-1)/2
    y = np.arange(N) - (N-1)/2

    # Create a 2D grid of coordinates
    X, Y = np.meshgrid(x, y)

    # Flatten the grids and combine the x, y coordinates into a list of tuples
    sq_points = list(zip(X.flatten(), Y.flatten()))

    return sq_points

def add_disorder(points_list, alpha=0.8, rseed=10):
    """Add small disorder to the coordinate positions."""

    np.random.seed(rseed)

    # Add small randomness to the points
    rnd = np.random.rand(len(points_list), 2) - 0.5
    points = points_list + alpha*rnd

    return points

def generate_voronoi_graph(voronoi):
    """Generate a Voronoi graph from a Voronoi diagram."""
    
    G = nx.Graph()
    for ridge in vor.ridge_vertices:
        # If the ridge is finite, add an edge
        if ridge[0] >= 0 and ridge[1] >= 0:
            G.add_edge(ridge[0], ridge[1])

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    
    pos = [list(vor.vertices[i]) for i in range(len(vor.vertices))]
    
    return H, pos

def prune_graph(graph, positions, radius=5):
    """Prune the graph to only inlcude nodes and edges within a specified area."""
    
    pos = {str(i): positions[i] for i in range(len(positions))}
    
    for node, point in enumerate(positions):
        if np.linalg.norm(point) > radius:
            graph.remove_node(node)
            pos.pop(str(node))
    
    #relabel the graph with consecutive integers
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes)}
    graph = nx.relabel_nodes(graph, mapping)
    
    new_pos = list(pos.values())

    return graph, new_pos

def plot_graph(vor, graph, positions):
    """Plot the graph and coordinate points."""
    
    # Plot the Voronoi diagram
    fig, ax = plt.subplots()

    # Plot the Voronoi diagram using scipy's built-in function
    #voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=0.5, line_alpha=0.5)

    # Plot the network graph on top
    nx.draw(
            graph, 
            positions, 
            node_size=1, 
            width=0.2,
            node_color='black', 
            with_labels=False, 
            edge_color='green',
            alpha=1
            )
    
    ax.set_xlim([-5.5,5.5])
    ax.set_ylim([-5.5,5.5])
    ax.set_aspect('equal', 'box')
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.title("Voronoi Network")
    plt.savefig("voronoi_graph.pdf")
    #plt.show()


##### START HERE ######
    
# Generate hexagonal grid points
hex_grid_points = generate_hex_grid()

# Generate square grid points
sq_grid_points = generate_square_grid()

# Add small random displacement to the points
points = add_disorder(hex_grid_points)

# Generate Voronoi diagram
vor = Voronoi(points)

G, pos = generate_voronoi_graph(vor)
H = G.copy()
H, pos2 = prune_graph(H, pos, 5)

#plot_graph(vor,H,pos2)

# save graph for netSALT
for u in H.nodes:
    H.nodes[u]['position'] = pos2[u]

save_graph(H, 'voronoi_graph.gpickle')