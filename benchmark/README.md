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

Workloads sized to put 22–50 modes inside the rectangle, so the
mode-count-vs-`probe_dim` ceiling actually bites. Beyn caps at
`probe_dim` modes per contour as a fundamental property of its
SVD-extraction step, and `find_modes_contour` clamps `probe_dim` to
the node count — so on a small graph (16/21 nodes) a wide rectangle
containing more modes than nodes makes the *single* contour
under-count to 0. Subdivision is what fixes that.

| workload                     | contour       | contour-subdiv  | grid+root              |
|------------------------------|--------------:|----------------:|-----------------------:|
| line n=15, 38 modes, k ≤ 60  | 197 ms (**0**) | 1576 ms (38)   | 25537 ms (36 — missed 2) |
| line n=20, 50 modes, k ≤ 80  | 212 ms (**0**) | 3039 ms (50)   | 35782 ms (48 — missed 2) |
| buffon ~60 nodes, 22 modes   | 298 ms (22)    | 1070 ms (22)   | 26492 ms (22)          |

Single contour wins handily on the buffon graph (60 nodes ≫ 22 modes,
plenty of headroom); subdivision is overhead. On the cramped
line graphs, single returns nothing useful and subdivision is
mandatory. `grid+root` is consistently 10–30× slower than the
contour methods and consistently misses a couple of modes that
`peak_local_max` merged at `min_distance=2`.

### Does positional accuracy require subdivision?

**No, not for accuracy** — only for *coverage*. Across all working
configurations (single contour on the buffon graph, subdivision on
the line graphs), positional error vs the gold reference is
`~1e-11`–`1e-8` regardless of `n_k`. Tightening the match tolerance
from `1e-2` to `1e-6` doesn't change which methods agree. The
subdivision sweep in `results_search.md` shows it directly: on
buffon, every `n_k ∈ {1, 2, 4, 8, 16}` returns the same 22 modes at
indistinguishable accuracy. On the line graphs, `n_k` controls
whether you find the modes at all; once you cross the
modes-per-cell ≤ `probe_dim` threshold (`n_k=4` for the n=15 case,
`n_k=8` for n=20), additional subdivision adds time without adding
information.

The grid path missed one mode on the buffon graph: a near-corner mode
that `peak_local_max` merged into a neighbour at `min_distance=2`.
Beyn caught it.

## Adding a workload

Add an entry to either script's `workloads`/`cases` list and pass a
graph constructed via `_common.line_graph`,
`_common.buffon_planar_graph`, or your own factory. The cross-method
consistency check is built into both scripts — a regression that
makes one method drift from the consensus will fail the run loudly.
