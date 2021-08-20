import numpy as np
from tqdm import tqdm
from scipy.linalg import expm, pinv


def hat(xi_vec):
    """Convert vector Lie algebra to matrix Lie algebra element."""
    xi = np.zeros((3, 3))
    xi[1, 2] = -xi_vec[0]
    xi[2, 1] = xi_vec[0]
    xi[0, 2] = xi_vec[1]
    xi[2, 0] = -xi_vec[1]
    xi[0, 1] = -xi_vec[2]
    xi[1, 0] = xi_vec[2]
    return xi


def hat_inv(xi):
    """Convert matrix Lie algebra to vector Lie algebra element."""
    xi_vec = np.zeros(3)
    xi_vec[0] = xi[2, 1]
    xi_vec[1] = xi[0, 2]
    xi_vec[2] = xi[1, 0]
    return xi_vec


def g(x, chi):
    """Exponential map between Lie algebra and group, or Adjoint action on vector Lie algebra."""
    return expm(chi * x)


def Ad(xi, x, chi):
    """Adjoint action on matrix Lie algebra."""
    _g = g(x, chi)
    return _g.dot(xi).dot(_g.T)


def W(x, chi):
    """Weight matrix diagonal element acting on vector Lie algebra."""
    return g(2 * x, chi) - np.eye(3)


def Winv(x, chi):
    """Inverse of weight matrix.

    We use pseudo inverse, so that the part along chi does not cause trouble.
    """
    return pinv(W(x, chi))


if __name__ == "__main__":
    chi = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 0]])  # along [0, 0, 1]
    xi_vec = [1, 1, 1]
    xi = hat(xi_vec)
    x = 2.2
    print(Winv(x, chi).dot(chi.dot(xi_vec)))
    print(chi.dot(Winv(x, chi).dot(xi_vec)))
    import netsalt
    import networkx as nx
    import matplotlib.pyplot as plt

    np.random.seed(42)
    graph = nx.grid_graph([3], periodic=True)
    graph = nx.convert_node_labels_to_integers(graph)
    netsalt.create_quantum_graph(graph, {"open_model": "closed"}, lengths=[0.8, 0.7, 1.2])
    for e in graph.edges:
        print(graph[e[0]][e[1]])
    netsalt.set_dispersion_relation(graph, netsalt.physics.dispersion_relation_linear, {"c": 1})
    freq = 10
    from scipy import sparse

    chi = hat([0.0, 0.0, 1.0])
    eps = 1.0
    c = [0.0, eps, 1.0 - eps]
    chi1 = hat(c / np.linalg.norm(c))
    c = [eps, 0.0, 1.0 - eps]
    chi2 = hat(c / np.linalg.norm(c))
    chis = len(graph.edges) * [chi]
    chis[-1] = chi2
    chis[-2] = chi1
    qs1 = []
    qs2 = []
    qs3 = []
    qs4 = []
    qs5 = []
    qs6 = []
    freqs = np.linspace(1.0, 20., 2000)
    for freq in tqdm(freqs):
        laplacian = netsalt.quantum_graph.construct_laplacian(freq, graph, group="SO3", chis=chis)
        eigs = np.sort(
            abs(
                sparse.linalg.eigsh(
                    laplacian,
                    k=6,
                    sigma=1e-2,
                    return_eigenvectors=False,
                    which="LM",
                    v0=np.ones(3 * len(graph)),
                )
            )
        )
        qs1.append(eigs[0])
        qs2.append(eigs[1])
        qs3.append(eigs[2])
        qs4.append(eigs[3])
        qs5.append(eigs[4])
        qs6.append(eigs[5])
    plt.figure()
    plt.semilogy(freqs, qs1, "-", lw=0.5, label="1")
    plt.semilogy(freqs, qs2, "-", lw=0.5, label="2")
    plt.semilogy(freqs, qs3, "-", lw=0.5, label="3")
    plt.semilogy(freqs, qs4, "-", lw=0.5, label="4")
    plt.semilogy(freqs, qs5, "-", lw=0.5, label="5")
    plt.semilogy(freqs, qs6, "-", lw=0.5, label="6")
    plt.legend()
    # plt.axis([freqs[0], freqs[-1], 1e-6, 1e2])
    plt.savefig("scan.pdf")
