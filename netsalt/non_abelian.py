"""Module for non-abelian quantum graphs."""
import numpy as np
from scipy import sparse, linalg

DIM = 3


def hat_inv(xi_vec):
    """Convert vector Lie algebra to matrix Lie algebra element."""
    xi = np.zeros((3, 3), dtype=np.complex128)
    xi[1, 2] = -xi_vec[0]
    xi[2, 1] = xi_vec[0]
    xi[0, 2] = xi_vec[1]
    xi[2, 0] = -xi_vec[1]
    xi[0, 1] = -xi_vec[2]
    xi[1, 0] = xi_vec[2]
    return xi


def hat(xi):
    """Convert matrix Lie algebra to vector Lie algebra element."""
    xi_vec = np.zeros(3, dtype=np.complex128)
    xi_vec[0] = xi[2, 1]
    xi_vec[1] = xi[0, 2]
    xi_vec[2] = xi[1, 0]
    return xi_vec


def proj_perp(chi_mat):
    """Perpendicular projection."""
    return chi_mat.dot(chi_mat.T) / norm(chi_mat) ** 2


def proj_paral(chi_mat):
    """Paralell projection."""
    return np.eye(3) - proj_perp(chi_mat)


def norm(chi_mat):
    """Norm of chi"""
    return np.sqrt(np.trace(0.5 * chi_mat.dot(chi_mat.T)))


def Ad(chi_mat):
    """Adjoint action."""
    return linalg.expm(chi_mat)


def set_so3_wavenumber(graph, wavenumber):
    """Set so3 matrix wavenumber."""
    chis = [graph[u][v].get("chi", None) for u, v in graph.edges]
    if chis[0] is None:
        chi = hat_inv(np.array([0.0, 0.0, 1.0]))
        chis = np.array(len(graph.edges) * [chi])
    else:
        if len(np.shape(chis[0])) == 1:
            chis = np.array([hat_inv(chi) for chi in chis])
    graph.graph["ks"] = chis * np.real(wavenumber) - np.eye(3) * np.imag(wavenumber)
    graph.graph["chis"] = chis
    graph.graph["wavenumber"] = wavenumber


def construct_so3_incidence_matrix(graph, abelian_scale=1.0):
    """Construct SO3 incidence matrix."""

    def _ext(i):
        return slice(DIM * i, DIM * (i + 1))

    B = sparse.lil_matrix((len(graph.edges) * 2 * DIM, len(graph) * DIM), dtype=np.complex128)
    BT = sparse.lil_matrix((len(graph) * DIM, len(graph.edges) * 2 * DIM), dtype=np.complex128)
    for ei, (u, v) in enumerate(graph.edges):
        one = np.eye(DIM)
        expl = Ad(graph.graph["lengths"][ei] * graph.graph["ks"][ei])
        expl = np.array(expl.dot(proj_perp(graph.graph["chis"][ei])), dtype=np.complex128)
        expl += (
            abelian_scale
            * np.exp(1.0j * graph.graph["lengths"][ei] * graph.graph["wavenumber"])
            * proj_paral(graph.graph["chis"][ei])
        )

        B[_ext(2 * ei), _ext(u)] = -one
        B[_ext(2 * ei), _ext(v)] = expl
        B[_ext(2 * ei + 1), _ext(u)] = expl
        B[_ext(2 * ei + 1), _ext(v)] = -one

        BT[_ext(u), _ext(2 * ei)] = -one
        BT[_ext(v), _ext(2 * ei)] = expl
        BT[_ext(u), _ext(2 * ei + 1)] = expl
        BT[_ext(v), _ext(2 * ei + 1)] = -one

        if graph.graph["params"]["open_model"] == "open":
            if len(graph[u]) == 1 or len(graph[v]) == 1:
                BT[_ext(v), _ext(2 * ei)] = 0
                BT[_ext(u), _ext(2 * ei + 1)] = 0

        if graph.graph["params"]["open_model"] == "directed":
            BT[_ext(u), _ext(2 * ei + 1)] = 0
            BT[_ext(v), _ext(2 * ei + 1)] = 0

        if graph.graph["params"]["open_model"] == "directed_reversed":
            B[_ext(2 * ei + 1), _ext(u)] = 0
            B[_ext(2 * ei + 1), _ext(v)] = 0

    return BT, B


def construct_so3_weight_matrix(graph, with_k=True, abelian_scale=1.0):
    """Construct SO3 weight matrix."""

    def _ext(i):
        return slice(DIM * i, DIM * (i + 1))

    Winv = sparse.lil_matrix(
        (len(graph.edges) * 2 * DIM, len(graph.edges) * 2 * DIM), dtype=np.complex128
    )
    for ei, _ in enumerate(graph.edges):
        k = graph.graph["ks"][ei]
        chi = graph.graph["chis"][ei]
        length = graph.graph["lengths"][ei]

        w_perp = Ad(2.0 * length * k).dot(proj_perp(chi))
        w_paral = (
            abelian_scale * np.exp(2.0j * length * graph.graph["wavenumber"]) * proj_paral(chi)
        )
        w = w_perp + w_paral - np.eye(3)

        winv = linalg.inv(w)

        if with_k:
            winv = (k.dot(proj_perp(chi)) + 1.0j * graph.graph["wavenumber"] * proj_paral(chi)).dot(
                winv
            )

        Winv[_ext(2 * ei), _ext(2 * ei)] = winv
        Winv[_ext(2 * ei + 1), _ext(2 * ei + 1)] = winv
    return Winv


def construct_so3_laplacian(wavenumber, graph, abelian_scale=1.0, with_k=True):
    """Construct quantum laplacian from a graph."""
    set_so3_wavenumber(graph, wavenumber)
    BT, B = construct_so3_incidence_matrix(graph, abelian_scale=abelian_scale)
    Winv = construct_so3_weight_matrix(graph, abelian_scale=abelian_scale, with_k=with_k)
    return BT.dot(Winv).dot(B), BT, B, Winv
