"""How many sub-intervals does an edge need, and where does that cost land?

Issue #52: spatial hole burning makes the permittivity vary *within* an edge,
which breaks the piecewise-constant assumption the quantum-graph secular matrix
is built on. Today the only remedy is ``oversample_graph`` -- subdivide until
eps is nearly constant on each sub-edge -- and that puts the sub-interval count
into the *eigenproblem*: the production buffon needs 76803 nodes against its 243
edges to reach lambda/12, which is what makes full SALT unaffordable there.

A per-edge transfer matrix keeps the eigenproblem at 243 edges and makes the
sub-interval count local, independent, and parallel. This script measures what
that count has to be, for the scheme oversampling implicitly uses and for a
fourth-order alternative.

The two things it establishes:

1. **Oversampling is second-order Magnus.** Freezing eps at each sub-edge
   midpoint and multiplying closed-form constant-eps propagators is *exactly*
   ``exp(h A(x_mid))`` for this system. The two columns agree to round-off, which
   is what makes the comparison below a fair one.
2. **Fourth order needs ~128x fewer sub-intervals** for the same accuracy. On top
   of moving the count out of the matrix, that is a second, independent factor.

Run from this directory::

    OMP_NUM_THREADS=1 python probe_edge_propagator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netsalt.edge_propagator import edge_transfer_matrix  # noqa: E402

#: A buffon-like edge: length ~11 at k = 10.7 and n = 1.5 holds ~28 oscillations,
#: which is exactly the regime where the default node cap aliases (#52).
LENGTH = 11.0
K = 10.7
EPS0 = 2.25
#: Saturation ripple depth. Hole burning is a few-percent effect on eps at the
#: pumps of interest, modulated at the standing-wave period.
RIPPLE = 0.04
Q0 = K * np.sqrt(EPS0)

TARGET = 1e-8


def saturated_eps(x):
    """eps0 / (1 + s cos^2(q x)) -- a standing-wave-modulated saturation."""
    return EPS0 / (1.0 + RIPPLE * np.cos(Q0 * np.asarray(x)) ** 2)


def reference_transfer():
    """Ground truth: integrate the ODE itself to near machine precision."""

    def rhs(x, y):
        e = saturated_eps(x)
        return [y[1], -(K**2) * e * y[0], y[3], -(K**2) * e * y[2]]

    solution = solve_ivp(
        rhs, (0.0, LENGTH), [1.0, 0.0, 0.0, 1.0], rtol=1e-13, atol=1e-14, method="DOP853"
    )
    y = solution.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def steps_for_target(method, reference, scale):
    n = 25
    while n <= 2_000_000:
        error = np.linalg.norm(
            edge_transfer_matrix(K, LENGTH, saturated_eps, n_steps=n, method=method) - reference
        )
        if error / scale < TARGET:
            return n
        n *= 2
    return None


def main():
    reference = reference_transfer()
    scale = np.linalg.norm(reference)
    oscillations = Q0 * LENGTH / (2 * np.pi)
    print(
        f"edge: length {LENGTH}, k {K}, n {np.sqrt(EPS0):.2f} "
        f"-> {oscillations:.1f} oscillations, {RIPPLE:.0%} saturation ripple"
    )
    print(f"reference from a DOP853 solve at rtol 1e-13 (norm {scale:.4g})\n")

    print(f"{'sub-intervals':>14} {'magnus2 (= oversampling)':>26} {'magnus4':>12}")
    for n in (25, 50, 100, 200, 400, 800, 1600):
        errors = [
            np.linalg.norm(
                edge_transfer_matrix(K, LENGTH, saturated_eps, n_steps=n, method=m) - reference
            )
            / scale
            for m in ("magnus2", "magnus4")
        ]
        print(f"{n:>14} {errors[0]:>26.3e} {errors[1]:>12.3e}")

    print(f"\nsub-intervals needed for {TARGET:g} relative error:")
    counts = {}
    for method in ("magnus2", "magnus4"):
        counts[method] = steps_for_target(method, reference, scale)
        print(f"  {method:9s} {counts[method]:>9d}")
    if counts["magnus2"] and counts["magnus4"]:
        print(f"  -> {counts['magnus2'] / counts['magnus4']:.0f}x fewer at fourth order")

    print(
        "\nWhere the count lands is the larger point. Today it multiplies the\n"
        "graph: 243 buffon edges x N sub-edges is the eigenproblem size, and\n"
        "N ~ 316 already means 76803 nodes. As a per-edge transfer matrix the\n"
        "eigenproblem stays 243 edges and N becomes local, independent work."
    )


if __name__ == "__main__":
    main()
