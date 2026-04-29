# Benchmarks

Reproducible micro-benchmarks for the mode-search and mode-refinement
algorithms shipped in this branch.

## Layout

- `_common.py` — graph constructors (line, buffon planar) and shared
  helpers (`mode_quality` call counter, wall-time context manager).
- `bench_refine.py` — compares the four refiners
  (`refine_mode_brownian_ratchet`, `refine_mode_root`,
  `refine_mode_newton`, `refine_mode_nelder_mead`) on multiple graphs,
  multiple modes per graph, and three perturbation magnitudes. Every
  converged refiner is cross-checked against the others — output is
  asserted to land within 5e-3 of the consensus, so the script also
  doubles as a regression suite.
- `bench_search.py` — compares full-rectangle mode searchers:
  `find_modes_contour`, `find_modes_contour_subdivided`, and the legacy
  grid-scan + `peak_local_max` + `refine_mode_root` pipeline. Reports
  modes-found, worst `|λ₁|`, and cross-method agreement counts.
- `results_refine.md`, `results_search.md` — captured results from a
  recent run on the developer's machine; regenerate via the commands
  below.

## Running

From the repo root, with a virtualenv that has `netsalt` installed
editable (`uv pip install -e .`):

```bash
.venv/bin/python benchmark/bench_refine.py --output benchmark/results_refine.md
.venv/bin/python benchmark/bench_search.py --output benchmark/results_search.md
```

Each script prints the same table it writes to disk; runs in well under
a minute on a recent laptop.

## Headline numbers

These are local-machine numbers (single-threaded, scipy 1.x, Python
3.12) — relative ratios are the interesting bit, absolute timings will
swing.

### Refinement

`root` and `newton` are 1–2 orders of magnitude faster than the
brownian ratchet, with `newton` typically using 1–2 `mode_quality`
calls vs. `root`'s 9–14 and brownian's 40–330. `nelder_mead` is the
slowest converger of the three modern methods (~60 evaluations) but
the only derivative-free one. See `results_refine.md` for the full
matrix; representative line graph numbers (n_edges=20, mode k≈π/2):

| perturbation | brownian | root | newton | nelder_mead |
|---|---:|---:|---:|---:|
| near (Δ=0.005) | 62 ms / 46 evals | 16 ms / 10 | 7 ms / 1 | 87 ms / 52 |
| mid  (Δ=0.02 ) | 124 ms / 77 | 38 ms / 21 | 13 ms / 2 | 96 ms / 55 |
| far  (Δ=0.05 ) | 394 ms / 231 | 18 ms / 11 | 13 ms / 2 | 106 ms / 63 |

### Mode search

Beyn (single contour) is ~25–40× faster than `grid+root` on these
workloads while landing every mode within `1e-2` of the
gold-reference subdivided-contour run. Subdivision adds overhead
without finding extra modes when the single contour's `probe_dim`
already exceeds the in-rectangle mode count, so for small graphs the
single contour wins.

| workload | contour | contour-subdiv | grid+root |
|---|---:|---:|---:|
| line n=15 | 104 ms (9 modes) | 191 ms (9) | 3843 ms (9) |
| line n=20 | 110 ms (9) | 193 ms (9) | 4152 ms (9) |
| buffon ~60 nodes | 222 ms (9) | 430 ms (9) | 4818 ms (8 — missed 1) |

The grid path missed one mode on the buffon graph: a near-corner mode
that `peak_local_max` merged into a neighbour at `min_distance=2`.
Beyn caught it.

## Adding a workload

Add an entry to either script's `workloads`/`cases` list and pass a
graph constructed via `_common.line_graph`,
`_common.buffon_planar_graph`, or your own factory. The cross-method
consistency check is built into both scripts — a regression that
makes one method drift from the consensus will fail the run loudly.
