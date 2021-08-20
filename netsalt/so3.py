import numpy as np
from tqdm import tqdm
from scipy.linalg import expm, pinv
import netsalt
import networkx as nx
import matplotlib.pyplot as plt


def hat_inv(xi_vec):
    """Convert vector Lie algebra to matrix Lie algebra element."""
    xi = np.zeros((3, 3))
    xi[1, 2] = -xi_vec[0]
    xi[2, 1] = xi_vec[0]
    xi[0, 2] = xi_vec[1]
    xi[2, 0] = -xi_vec[1]
    xi[0, 1] = -xi_vec[2]
    xi[1, 0] = xi_vec[2]
    return xi


def hat(xi):
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


def proj_perp(chi):
    return np.eye(3) - proj_paral(chi)


def proj_paral(chi):
    chi_h = hat(chi)
    return np.outer(chi_h, chi_h) / np.linalg.norm(chi_h) ** 2


if __name__ == "__main__":

    x = 2.2

    np.random.seed(42)
    graph = nx.grid_graph([3], periodic=True)
    graph = nx.convert_node_labels_to_integers(graph)
    netsalt.create_quantum_graph(graph, {"open_model": "closed"}, lengths=[0.8, 1.2, 1.0])
    graph.graph["params"]["c"] = 1
    netsalt.set_dispersion_relation(graph, netsalt.physics.dispersion_relation_linear)

    freq = 10
    from scipy import sparse

    chi = hat_inv([0, 0, 1.0])
    eps = 0.2
    c = [0.0, eps, 1.0 - eps]
    chi1 = hat_inv(c / np.linalg.norm(c))
    c = [eps, 0.0, 1.0 - eps]
    chi2 = hat_inv(c / np.linalg.norm(c))
    chis = len(graph.edges) * [chi]
    chis[-1] = chi2
    chis[-2] = chi1
    qs1 = []
    qs2 = []
    qs3 = []
    qs4 = []
    qs5 = []
    qs6 = []
    qsa = []
    freqs = np.linspace(1.0, 10.0, 2000)

    laplacian = netsalt.quantum_graph.construct_laplacian(
        2 * np.pi / 3.0, graph, group="SO3", chis=chis
    )
    e, v = sparse.linalg.eigs(
        laplacian,
        k=6,
        sigma=1e-2,
        which="LM",
        v0=np.ones(3 * len(graph)),
    )
    print(laplacian.dot(v[:, 2]), e)
    for freq in tqdm(freqs):

        ab_laplacian = netsalt.quantum_graph.construct_laplacian(freq, graph)
        eig = abs(
            sparse.linalg.eigs(
                ab_laplacian,
                k=1,
                sigma=1e-2,
                return_eigenvectors=False,
                which="LM",
                v0=np.ones(len(graph)),
            )
        )[0]
        qsa.append(eig)
        laplacian = netsalt.quantum_graph.construct_laplacian(freq, graph, group="SO3", chis=chis)
        eigs = np.sort(
            abs(
                sparse.linalg.eigs(
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

    plt.axvline(2 * np.pi / 3, c="k")
    plt.axvline(4 * np.pi / 3, c="k")
    plt.axvline(6 * np.pi / 3, c="k")
    plt.axvline(8 * np.pi / 3, c="k")
    plt.semilogy(freqs, qsa, "r-", lw=1.0, label="ab")
    plt.semilogy(freqs, qs1, "-", lw=0.5, label="1")
    plt.semilogy(freqs, qs2, "-", lw=0.5, label="2")
    plt.semilogy(freqs, qs3, "-", lw=0.5, label="3")
    plt.semilogy(freqs, qs4, "-", lw=0.5, label="4")
    plt.semilogy(freqs, qs5, "-", lw=0.5, label="5")
    plt.semilogy(freqs, qs6, "-", lw=0.5, label="6")
    plt.legend()
    # plt.axis([freqs[0], freqs[-1], 1e-6, 1e2])
    plt.savefig("scan.pdf")
