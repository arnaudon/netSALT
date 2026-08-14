"""A completely independent SALT solver for a 1D open cavity.

No netsalt code is imported here.  Two independent engines:

**(1) Transfer matrix / shooting** -- exact for piecewise-constant media.
Used for the passive modes and the (unsaturated) lasing thresholds, where the
medium *is* piecewise constant, so these answers carry no discretisation error
at all beyond the root finder's tolerance.

**(2) Finite differences + damped Newton** -- the full multimode SALT solve.
For lasing modes ``mu = 1..M`` we solve, on the grid ``x_i = i h``,
``h = L/N``,

    (D2 phi_mu)_i + k_mu^2 eps_eff[mu,i] phi_mu,i = 0,      i = 0..N
    sum_i w_i |phi_mu,i|^2 = 1                      (normalisation)
    Im phi_mu,i0 = 0                                (phase fixing)

    eps_eff[mu,i] = eps(x_i) + gamma(k_mu) D0(x_i) / (1 + sum_nu Gamma_nu a_nu |phi_nu,i|^2)

with ``Gamma_nu = -Im gamma(k_nu) = |gamma(k_nu)|^2`` and the physical field
``Psi_mu = sqrt(a_mu) phi_mu``, i.e. ``a_mu = int_cavity |Psi_mu|^2 dx``.
Unknowns per mode: ``(Re phi, Im phi, a_mu, k_mu)`` -- ``2(N+1)+2`` reals, and
the same number of equations, so the system is square and Newton applies
directly.  The explicit ``a_mu`` + normalisation removes the trivial
``phi = 0`` root that a bare ``A(k,|Psi|^2) Psi = 0`` formulation always has.

``D2`` is the standard second-order three-point Laplacian with the outgoing
(Robin) condition ``phi'(0) = -i k phi(0)``, ``phi'(L) = +i k phi(L)`` imposed
through symmetric ghost points, so the whole scheme is O(h^2) and Richardson
extrapolation in ``h^2`` is meaningful.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import _numdiff, root


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------
def gamma(k, k_a, gamma_perp):
    """Lorentzian gain gamma(k) = gp / (Re k - k_a + i gp)."""
    return gamma_perp / (np.real(k) - k_a + 1.0j * gamma_perp)


def gamma_clamp(k, k_a, gamma_perp):
    """Saturation factor Gamma = -Im gamma = |gamma|^2 (real k)."""
    return -np.imag(gamma(k, k_a, gamma_perp))


# --------------------------------------------------------------------------
# (1) transfer matrix / shooting -- exact for piecewise-constant media
# --------------------------------------------------------------------------
def secular_tm(k, widths, eps_slabs):
    """Outgoing-BC secular function of a stack of slabs embedded in vacuum.

    Shoots ``(psi, psi') = (1, -i k)`` from ``x = 0`` (purely outgoing to the
    left) and returns ``psi'(L) - i k psi(L)``; its zeros in complex ``k`` are
    the modes.  Exact within each slab (no discretisation).
    """
    k = complex(k)
    y = np.array([1.0 + 0j, -1j * k])
    for h, eps in zip(widths, eps_slabs, strict=True):
        q = k * np.sqrt(complex(eps))
        if abs(q * h) < 1e-14:
            t = np.array([[1.0 + 0j, h + 0j], [0.0 + 0j, 1.0 + 0j]])
        else:
            c, s = np.cos(q * h), np.sin(q * h)
            t = np.array([[c, s / q], [-q * s, c]])
        y = t @ y
    return y[1] - 1j * k * y[0]


def field_tm(k, widths, eps_slabs, x):
    """Shooting solution psi(x) (psi(0)=1, outgoing to the left) sampled at x."""
    k = complex(k)
    y = np.array([1.0 + 0j, -1j * k])
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    out = np.zeros(len(x), dtype=complex)
    for j, (h, eps) in enumerate(zip(widths, eps_slabs, strict=True)):
        q = k * np.sqrt(complex(eps))
        sel = (x >= edges[j] - 1e-15) & (x <= edges[j + 1] + 1e-15)
        d = x[sel] - edges[j]
        out[sel] = np.cos(q * d) * y[0] + np.sin(q * d) / q * y[1]
        c, s = np.cos(q * h), np.sin(q * h)
        y = np.array([[c, s / q], [-q * s, c]]) @ y
    return out


def passive_modes_tm(k_guesses, eps, length):
    """Complex passive modes of a uniform slab, by root finding on ``secular_tm``."""
    out = []
    for k0 in k_guesses:

        def res(v, _k0=k0):
            f = secular_tm(v[0] + 1j * v[1], [length], [eps])
            return [f.real, f.imag]

        sol = root(res, [np.real(k0), np.imag(k0)], method="hybr", tol=1e-14)
        out.append(sol.x[0] + 1j * sol.x[1])
    return np.array(out)


def threshold_tm(k0, eps, length, k_a, gamma_perp, d0_guess=0.5, pump=1.0, widths=None):
    """Non-interacting lasing threshold: real ``k`` and ``D0`` with a real-k root.

    Solves ``secular_tm(k; eps + gamma(k) D0 pump) = 0`` for the two real
    unknowns ``(k, D0)``.  Exact (uniform, unsaturated medium).
    """

    w = np.atleast_1d(np.asarray([length], dtype=float) if widths is None else widths)
    pm = np.broadcast_to(np.asarray(pump, dtype=float), w.shape)

    def res(v):
        k, d0 = float(v[0]), float(v[1])
        e = eps + gamma(k, k_a, gamma_perp) * d0 * pm
        f = secular_tm(k, w, e)
        return [f.real, f.imag]

    sol = root(res, [float(np.real(k0)), float(d0_guess)], method="hybr", tol=1e-14)
    return float(sol.x[0]), float(sol.x[1]), np.max(np.abs(res(sol.x)))


# --------------------------------------------------------------------------
# (2) finite-difference full SALT
# --------------------------------------------------------------------------
class FDSalt:
    """Finite-difference multimode SALT solver on ``[0, L]``."""

    def __init__(self, length, eps, n_grid, k_a, gamma_perp, pump=None):
        self.L = float(length)
        self.N = int(n_grid)
        self.h = self.L / self.N
        self.x = np.linspace(0.0, self.L, self.N + 1)
        self.eps = np.full(self.N + 1, float(eps), dtype=complex)
        self.k_a = float(k_a)
        self.gp = float(gamma_perp)
        self.pump = np.ones(self.N + 1) if pump is None else np.asarray(pump, dtype=float)
        # trapezoid weights for int_0^L . dx
        w = np.full(self.N + 1, self.h)
        w[0] = w[-1] = 0.5 * self.h
        self.w = w
        # k-independent part of the 3-point laplacian
        h2 = self.h**2
        main = np.full(self.N + 1, -2.0 / h2)
        off = np.full(self.N, 1.0 / h2)
        self._d2 = sp.diags([off, main, off], [-1, 0, 1], format="lil")
        self._d2[0, 1] = 2.0 / h2
        self._d2[self.N, self.N - 1] = 2.0 / h2
        self._d2 = sp.csr_matrix(self._d2, dtype=complex)

    # -- residual --------------------------------------------------------
    def _unpack(self, z, m):
        n = self.N + 1
        blk = 2 * n + 2
        phis, amps, ks = [], [], []
        for mu in range(m):
            b = z[mu * blk : (mu + 1) * blk]
            phis.append(b[:n] + 1j * b[n : 2 * n])
            amps.append(b[2 * n])
            ks.append(b[2 * n + 1])
        return phis, np.array(amps), np.array(ks)

    def pack(self, phis, amps, ks):
        out = []
        for phi, a, k in zip(phis, amps, ks, strict=True):
            out.append(np.concatenate([phi.real, phi.imag, [a], [k]]))
        return np.concatenate(out)

    def saturation_denom(self, phis, amps, ks):
        """1 + sum_nu Gamma_nu a_nu |phi_nu|^2, on the grid."""
        s = np.ones(self.N + 1)
        for phi, a, k in zip(phis, amps, ks, strict=True):
            s = s + gamma_clamp(k, self.k_a, self.gp) * a * np.abs(phi) ** 2
        return s

    def residual(self, z, m, d0, i0):
        phis, amps, ks = self._unpack(z, m)
        s = self.saturation_denom(phis, amps, ks)
        out = []
        for mu in range(m):
            k = ks[mu]
            eps_eff = self.eps + gamma(k, self.k_a, self.gp) * d0 * self.pump / s
            r = self._d2.dot(phis[mu])
            # Robin (outgoing) boundary rows carry the k-dependent ghost term
            r[0] += (2.0j * k / self.h) * phis[mu][0]
            r[-1] += (2.0j * k / self.h) * phis[mu][-1]
            r = r + k**2 * eps_eff * phis[mu]
            norm = float(np.sum(self.w * np.abs(phis[mu]) ** 2)) - 1.0
            out.append(np.concatenate([r.real, r.imag, [norm], [phis[mu].imag[i0[mu]]]]))
        return np.concatenate(out)

    # -- jacobian sparsity -----------------------------------------------
    def sparsity(self, m):
        n = self.N + 1
        blk = 2 * n + 2
        size = m * blk
        rows, cols = [], []

        def add(r, c):
            rows.append(r)
            cols.append(c)

        for mu in range(m):
            rb = mu * blk
            for i in range(n):
                # field rows of mode mu: own tridiagonal (Re & Im parts)
                for j in (i - 1, i, i + 1):
                    if 0 <= j <= self.N:
                        for cb, off in ((mu * blk, 0), (mu * blk, n)):
                            add(rb + i, cb + off + j)
                            add(rb + n + i, cb + off + j)
                # diagonal coupling to every mode's field at the same site
                for nu in range(m):
                    cb = nu * blk
                    for off in (0, n):
                        add(rb + i, cb + off + i)
                        add(rb + n + i, cb + off + i)
                    add(rb + i, cb + 2 * n)  # a_nu
                    add(rb + i, cb + 2 * n + 1)  # k_nu
                    add(rb + n + i, cb + 2 * n)
                    add(rb + n + i, cb + 2 * n + 1)
            # normalisation row: own field only
            for j in range(n):
                add(rb + 2 * n, mu * blk + j)
                add(rb + 2 * n, mu * blk + n + j)
            # phase row: one entry
            add(rb + 2 * n + 1, mu * blk + n)  # placeholder, filled below
        pat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size), dtype=np.int8)
        pat.data[:] = 1
        # phase rows: mark the whole Im block (cheap, keeps colouring valid)
        pat = sp.lil_matrix(pat)
        for mu in range(m):
            rb = mu * blk
            for j in range(n):
                pat[rb + 2 * n + 1, mu * blk + n + j] = 1
        return sp.csr_matrix(pat)

    # -- solve -----------------------------------------------------------
    def _pattern(self, m):
        cache = getattr(self, "_pat_cache", None)
        if cache is None:
            cache = self._pat_cache = {}
        if m not in cache:
            pat = self.sparsity(m)
            cache[m] = (pat, _numdiff.group_columns(pat))
        return cache[m]

    def solve(self, z0, m, d0, i0, tol=1e-10, max_iter=60, verbose=False):
        pat, groups = self._pattern(m)
        z = np.array(z0, dtype=float)
        r = self.residual(z, m, d0, i0)
        nrm = np.linalg.norm(r)
        for it in range(max_iter):
            if nrm < tol:
                break
            jac = _numdiff.approx_derivative(
                lambda v: self.residual(v, m, d0, i0),
                z,
                method="2-point",
                sparsity=(pat, groups),
            )
            try:
                dz = spla.spsolve(sp.csc_matrix(jac), -r)
            except Exception:  # pragma: no cover
                break
            if not np.all(np.isfinite(dz)):
                break
            step = 1.0
            for _ in range(30):
                zt = z + step * dz
                rt = self.residual(zt, m, d0, i0)
                if np.linalg.norm(rt) < nrm:
                    break
                step *= 0.5
            else:
                break
            z, r, nrm = zt, rt, np.linalg.norm(rt)
            if verbose:
                print(f"    newton {it}: |R| = {nrm:.3e} step={step}")
        phis, amps, ks = self._unpack(z, m)
        return z, phis, amps, ks, nrm

    # -- linear (unsaturated) secular function of the SAME discretisation --
    def secular_fd(self, k, d0=0.0):
        """Forward recursion through the FD scheme; zero <=> ``k`` is an FD mode.

        Uses exactly the operator :meth:`residual` builds, so comparing its
        roots against :func:`secular_tm` measures this scheme's O(h^2) error.
        """
        k = complex(k)
        eps_eff = self.eps + gamma(k, self.k_a, self.gp) * d0 * self.pump
        h2 = self.h**2
        p_prev = 1.0 + 0j  # phi_0
        p_cur = -(h2 / 2.0) * (-2.0 / h2 + 2.0j * k / self.h + k**2 * eps_eff[0]) * p_prev
        for i in range(1, self.N):
            nxt = 2.0 * p_cur - p_prev - h2 * k**2 * eps_eff[i] * p_cur
            p_prev, p_cur = p_cur, nxt
        return (2.0 / h2) * p_prev + (
            -2.0 / h2 + 2.0j * k / self.h + k**2 * eps_eff[self.N]
        ) * p_cur

    def passive_mode_fd(self, k0):
        def res(v):
            f = self.secular_fd(v[0] + 1j * v[1])
            return [f.real, f.imag]

        s = root(res, [np.real(k0), np.imag(k0)], method="hybr", tol=1e-14)
        return s.x[0] + 1j * s.x[1]

    def threshold_fd(self, k0, d0_guess=0.5):
        def res(v):
            f = self.secular_fd(float(v[0]), d0=float(v[1]))
            return [f.real, f.imag]

        s = root(res, [float(np.real(k0)), float(d0_guess)], method="hybr", tol=1e-14)
        return float(s.x[0]), float(s.x[1])

    # -- helpers ---------------------------------------------------------
    def seed_from_tm(self, k, eps_profile_fn, amp):
        """Initial (phi, a, k) from an exact shooting solution at wavenumber k."""
        eps_slabs = eps_profile_fn(k)
        widths = np.full(len(eps_slabs), self.L / len(eps_slabs))
        psi = field_tm(k, widths, eps_slabs, self.x)
        nrm = np.sqrt(np.sum(self.w * np.abs(psi) ** 2))
        phi = psi / nrm
        # rotate global phase so that phi is real at its largest-|phi| point
        i0 = int(np.argmax(np.abs(phi)))
        phi = phi * np.exp(-1j * np.angle(phi[i0]))
        return phi, amp, k, i0
