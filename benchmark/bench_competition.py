"""Mode-competition matrix benchmark: scalar loop vs vectorised kernel.

``compute_mode_competition_matrix`` builds an ``M x M`` matrix whose every
element used to be a pure-Python loop over all ``E`` edges, fanned out over a
multiprocessing pool. The vectorised kernel contracts the whole
``(mu, nu, edge)`` tensor with array ops in a single process.

Three costs are compared, all on synthetic per-mode data of the same shape the
pipeline produces:

``loop-serial``
    ``_compute_mode_competition_element_reference`` (the original scalar
    edge loop), called ``M*M`` times in one process. This is the honest
    per-CPU cost.
``loop-pool``
    the same, fanned out over ``n_workers`` processes, i.e. what the old
    ``compute_mode_competition_matrix`` actually did in wall time.
``batched``
    ``_compute_mode_competition_matrix_batched``, the new single-process path.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python benchmark/bench_competition.py
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from functools import partial
from time import perf_counter

import numpy as np

from netsalt.modes import (
    _compute_mode_competition_element_reference,
    _compute_mode_competition_matrix_batched,
)

EDGE_COUNTS = (250, 500, 2500)
MODE_COUNTS = (20, 50, 100)


def synthetic_case(n_modes, n_edges, seed=0):
    """Per-mode ``(ks, edge_flux, gamma)`` triples with pipeline-realistic scales."""
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(0.5, 1.5, n_edges)
    params = {
        "pump": np.ones(n_edges),
        "inner": np.ones(n_edges, dtype=bool),
    }
    # A couple of unpumped / outer edges, as a real graph has.
    params["pump"][: max(1, n_edges // 50)] = 0.0
    params["inner"][-max(1, n_edges // 50) :] = False

    precomp = []
    for m in range(n_modes):
        k0 = 10.0 + 0.05 * m - 1j * rng.uniform(0.001, 0.01)
        ks = k0 * rng.uniform(0.9, 1.1, n_edges)
        flux = rng.normal(size=2 * n_edges) + 1j * rng.normal(size=2 * n_edges)
        precomp.append((ks, flux, rng.normal() + 1j * rng.normal()))
    return lengths, params, precomp


def time_loop_serial(lengths, params, precomp, repeats=1):
    n = len(precomp)
    start = perf_counter()
    for _ in range(repeats):
        out = np.empty((n, n), dtype=np.complex128)
        for mu in range(n):
            for nu in range(n):
                out[mu, nu] = _compute_mode_competition_element_reference(
                    lengths, params, [precomp[mu][:2], precomp[nu][:2], precomp[nu][2]]
                )
    return (perf_counter() - start) / repeats, out


def time_loop_pool(lengths, params, precomp, n_workers):
    n = len(precomp)
    input_data = [
        [precomp[mu][:2], precomp[nu][:2], precomp[nu][2]] for mu in range(n) for nu in range(n)
    ]
    chunksize = max(1, int(0.1 * len(input_data) / n_workers))
    start = perf_counter()
    with multiprocessing.Pool(n_workers) as pool:
        out = list(
            pool.imap(
                partial(_compute_mode_competition_element_reference, lengths, params),
                input_data,
                chunksize=chunksize,
            )
        )
    return perf_counter() - start, np.asarray(out).reshape(n, n)


def time_batched(lengths, params, precomp, repeats=3):
    _compute_mode_competition_matrix_batched(lengths, params, precomp)  # warm up
    start = perf_counter()
    for _ in range(repeats):
        out = _compute_mode_competition_matrix_batched(lengths, params, precomp)
    return (perf_counter() - start) / repeats, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-pool", action="store_true", help="skip the multiprocessing column")
    args = parser.parse_args()

    header = (
        f"{'E':>6} {'M':>5} {'loop-serial':>12} {'loop-pool':>11} {'batched':>10} "
        f"{'speedup/serial':>15} {'speedup/pool':>13} {'max rel err':>12}"
    )
    print(f"n_workers = {args.workers}\n")
    print(header)
    print("-" * len(header))

    for n_edges in EDGE_COUNTS:
        for n_modes in MODE_COUNTS:
            lengths, params, precomp = synthetic_case(n_modes, n_edges)
            t_serial, ref = time_loop_serial(lengths, params, precomp)
            t_batched, new = time_batched(lengths, params, precomp)
            if args.no_pool:
                t_pool = float("nan")
            else:
                t_pool, pool_out = time_loop_pool(lengths, params, precomp, args.workers)
                assert np.array_equal(pool_out, ref)

            rel = np.abs(new - ref) / np.abs(ref)
            print(
                f"{n_edges:6d} {n_modes:5d} {t_serial:11.3f}s {t_pool:10.3f}s "
                f"{t_batched:9.4f}s {t_serial / t_batched:14.1f}x "
                f"{t_pool / t_batched:12.1f}x {rel.max():12.2e}"
            )


if __name__ == "__main__":
    main()
