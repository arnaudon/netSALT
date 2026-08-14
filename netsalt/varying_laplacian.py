r"""Secular matrix for edges whose permittivity varies *along* the edge.

:func:`~netsalt.quantum_graph.construct_laplacian` builds
:math:`L = B^T W^{-1} B` from the closed-form constant-:math:`\epsilon` edge
solution. That closed form is what makes the quantum-graph method exact, and it
is also why spatial hole burning currently forces ``oversample_graph``: a
saturated :math:`\epsilon` varies within an edge, so the edge must be chopped
until it is piecewise constant again -- which grows the *eigenproblem* (issues
#52, #53).

It does not have to. :math:`L` is a sum of independent per-edge :math:`2\times 2`
blocks, each determined by that edge's Dirichlet-to-Neumann map, and the DtN map
of a varying-:math:`\epsilon` edge is available from its transfer matrix. So the
edge count -- and hence the matrix size -- can stay fixed while the within-edge
resolution moves into local, parallel work.

**The block.** With :math:`M` the transfer matrix taking
:math:`(\psi, \psi')` from one end of the edge to the other and
:math:`\det M = 1` (Wronskian),

.. math::

    \psi_v = M_{11}\psi_u + M_{12}\psi'_u
    \;\Longrightarrow\;
    \psi'_u = \frac{\psi_v - M_{11}\psi_u}{M_{12}},

and eliminating :math:`\psi'_u` from :math:`\psi'_v = M_{21}\psi_u + M_{22}\psi'_u`
using :math:`M_{21} - M_{22}M_{11}/M_{12} = -\det M / M_{12} = -1/M_{12}` gives
the outgoing-derivative map

.. math::

    D = \frac{1}{M_{12}}\begin{pmatrix} M_{11} & -1 \\ -1 & M_{22}\end{pmatrix},
    \qquad L_{\rm edge} = -i\,D .

For constant :math:`\epsilon`,
:math:`M = \bigl(\begin{smallmatrix}\cos ql & \sin(ql)/q\\ -q\sin ql & \cos ql\end{smallmatrix}\bigr)`
recovers :math:`-i\,q\cot(ql)` on the diagonal and :math:`i\,q/\sin(ql)` off it --
exactly what ``construct_laplacian`` produces, which
``tests/test_unit.py::TestVaryingLaplacian`` asserts to 1e-12 on closed and open
graphs.

**Boundary edges.** Under ``open_model="open"`` an edge touching a degree-one
node carries the outgoing-wave condition, which ``construct_incidence_matrix``
implements by zeroing cross terms in :math:`B^T`; its block is
:math:`\frac{q}{e^2-1}\bigl(\begin{smallmatrix}1 & -e\\ -e & 1\end{smallmatrix}\bigr)`
with :math:`e = e^{iql}`. Those edges are the passive leads -- they carry no pump,
so hole burning leaves their :math:`\epsilon` constant and the closed form stays
exact. :func:`construct_laplacian_varying` therefore keeps it for them, and
raises if a boundary edge is handed a varying profile rather than silently
returning a number that is not the DtN map of anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy as sc

from .edge_propagator import edge_transfer_matrix, propagator_constant_eps
from .quantum_graph import set_wavenumber

__all__ = ["construct_laplacian_varying", "edge_dtn_block"]

#: Boundary models whose per-edge block this module knows how to assemble.
_SUPPORTED_MODELS = ("open", "closed", "custom")


def edge_dtn_block(transfer: np.ndarray) -> np.ndarray:
    r"""Kirchhoff block :math:`-i D` of an edge, from its transfer matrix.

    Args:
        transfer: the edge's 2x2 transfer matrix.

    Returns:
        The 2x2 contribution this edge makes to the secular matrix, in the
        ``(u, v)`` node ordering of the edge.

    Raises:
        ZeroDivisionError-like ``FloatingPointError``: not raised; a vanishing
            ``M[0, 1]`` is the edge sitting on its Dirichlet resonance (the
            ``k l in pi Z`` pole of issue #45), and is left to propagate as an
            inf/nan exactly as the closed-form path does.
    """
    m12 = transfer[0, 1]
    return -1j * np.array([[transfer[0, 0], -1.0], [-1.0, transfer[1, 1]]], dtype=complex) / m12


def _boundary_block(q: complex, length: float) -> np.ndarray:
    """Outgoing-wave block for an edge touching a degree-one node."""
    e = np.exp(1j * q * length)
    return (q / (e**2 - 1.0)) * np.array([[1.0, -e], [-e, 1.0]], dtype=complex)


def construct_laplacian_varying(
    wavenumber: complex,
    graph,
    eps_profiles: Sequence[Callable[[np.ndarray], np.ndarray] | None] | None = None,
    n_steps: int = 64,
    method: str = "magnus4",
):
    r"""Secular matrix allowing permittivity to vary within each edge.

    Args:
        wavenumber: vacuum wavenumber :math:`k`.
        graph: quantum graph. ``set_wavenumber`` is applied internally, as
            :func:`~netsalt.quantum_graph.construct_laplacian` does, so the
            per-edge ``ks`` used by constant-eps edges come from the graph's own
            dispersion relation at this ``wavenumber``.
        eps_profiles: one entry per edge, in ``graph.edges`` order. ``None``
            means "constant" and uses the edge's existing ``ks`` entry -- the
            exact closed form, no discretisation. A callable maps positions in
            ``[0, length]`` to permittivities.
        n_steps: sub-intervals per varying edge. Local to the edge; it does not
            enter the matrix, which is the point.
        method: passed to :func:`~netsalt.edge_propagator.edge_transfer_matrix`.

    Returns:
        The ``n x n`` secular matrix as a sparse CSC matrix, matching
        :func:`~netsalt.quantum_graph.construct_laplacian` exactly when every
        profile is ``None``.

    Raises:
        ValueError: if ``open_model`` is one of the directed variants (whose
            block structure this module does not implement), if ``eps_profiles``
            has the wrong length, or if a boundary edge is given a varying
            profile.
    """
    set_wavenumber(graph, wavenumber)
    params = graph.graph["params"]
    open_model = params["open_model"]
    if open_model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"construct_laplacian_varying does not implement open_model={open_model!r}; "
            f"supported: {_SUPPORTED_MODELS}. The directed models zero a different set "
            "of B/B^T entries, so their per-edge block is not the DtN map assembled here."
        )

    edges = list(graph.edges)
    n_edges = len(edges)
    if eps_profiles is None:
        eps_profiles = [None] * n_edges
    if len(eps_profiles) != n_edges:
        raise ValueError(f"eps_profiles has {len(eps_profiles)} entries for {n_edges} edges.")

    lengths = np.asarray(graph.graph["lengths"], dtype=float)
    ks = np.asarray(graph.graph["ks"])
    degrees = dict(graph.degree())
    outer_edges = params.get("outer_edges") if open_model == "custom" else None

    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for edge_index, (u, v) in enumerate(edges):
        length = float(lengths[edge_index])
        profile = eps_profiles[edge_index]

        if open_model == "open":
            is_boundary = degrees[u] == 1 or degrees[v] == 1
        elif open_model == "custom":
            is_boundary = outer_edges is not None and (u, v) in outer_edges
        else:
            is_boundary = False

        if is_boundary:
            if profile is not None:
                raise ValueError(
                    f"Edge {(u, v)} touches the open boundary and was given a varying "
                    "permittivity profile. The outgoing-wave block is only valid for "
                    "constant eps; boundary edges are passive leads (pump = 0), so "
                    "hole burning should leave them uniform. Pass None for them."
                )
            block = _boundary_block(ks[edge_index], length)
        elif profile is None:
            block = edge_dtn_block(propagator_constant_eps(ks[edge_index], length))
        else:
            transfer = edge_transfer_matrix(
                wavenumber, length, profile, n_steps=n_steps, method=method
            )
            block = edge_dtn_block(transfer)

        iu, iv = index[u], index[v]
        rows.extend((iu, iu, iv, iv))
        cols.extend((iu, iv, iu, iv))
        data.extend((block[0, 0], block[0, 1], block[1, 0], block[1, 1]))

    return sc.sparse.csc_matrix((np.asarray(data, dtype=complex), (rows, cols)), shape=(size, size))
