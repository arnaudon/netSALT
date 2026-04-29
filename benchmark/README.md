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
  grid-scan + `peak_local_max` + `refine_mode_root` pipeline on
  workloads of 22–50 modes. Includes a subdivision sweep that
  isolates how `n_k` interacts with `probe_dim`.
- `bench_stress.py` — high-mode-count stress test (~300 modes on a
  61-node buffon graph) that isolates the *two* mechanisms by which
  subdivision helps Beyn: the probe-dim ceiling and the trapezoidal
  quadrature accuracy. See "What does subdivision add?" below.
- `bench_tune_then_batch.py` — demonstrates the
  `tune_contour_parameters` workflow: run the adaptive search once on
  a representative graph, splat the returned parameter dict into
  `find_modes_contour_subdivided` for the rest of the batch.
- `results_refine.md`, `results_search.md`, `results_stress.md`,
  `results_tune_then_batch.md` — captured results from a recent run;
  regenerate via the commands below.

## Running

From the repo root, with a virtualenv that has `netsalt` installed
editable (`uv pip install -e .`):

```bash
.venv/bin/python benchmark/bench_refine.py --output benchmark/results_refine.md
.venv/bin/python benchmark/bench_search.py --output benchmark/results_search.md
.venv/bin/python benchmark/bench_stress.py --output benchmark/results_stress.md
.venv/bin/python benchmark/bench_tune_then_batch.py --output benchmark/results_tune_then_batch.md
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

### What does subdivision actually add?

Subdivision (`find_modes_contour_subdivided`) splits the scan
rectangle into `n_k × n_alpha` cells, runs `find_modes_contour` on
each independently, and concatenates the results with a
boundary-distance dedup. It helps for **two distinct reasons**, both
of which kick in at high mode count:

1. **The `probe_dim` ceiling.** The SVD of `A_0` has at most
   `probe_dim` non-zero singular values; modes beyond that are
   rank-deficient and dropped. `find_modes_contour` clamps
   `probe_dim` to the graph's node count, so on small graphs this
   ceiling is hard. Subdivision splits a rectangle with too many
   modes into cells where each cell's mode count fits under
   `probe_dim`.

2. **Trapezoidal-quadrature accuracy.** Even when `probe_dim`
   nominally exceeds the mode count, the moment integrals
   `A_j = ∮ kʲ · L⁻¹(k) · V dk` lose accuracy when many poles sit
   inside one large contour: the integrand varies rapidly near each
   pole and the smaller singular values get cut by `svd_tol`.
   Subdivision keeps the per-cell pole count small, so a modest
   `n_quad` (say 200) suffices.

`results_stress.md` isolates both effects on a buffon graph with 303
modes (61 nodes, `probe_dim` capped at 60):

* **Sweep 1** — fixed `n_quad=200`, vary `n_k`. `n_k=1, 2` return 0
  modes; `n_k=4, 6` partial; `n_k=8` recovers all 303 in 2.9 s. The
  transition lines up with `modes_per_cell ≤ 0.65 · probe_dim`.
* **Sweep 2** — same rectangle on a *denser* graph (340 nodes,
  254 modes, `probe_dim=340` so the ceiling is out of the way).
  Single contour at `n_quad=200` finds 4 modes; at `n_quad=1600` it
  finds 215 in 17 s, still missing ~15%. Subdivision is the cheaper
  route at any mode count this high.

**Rule of thumb:** pick `n_k` so that
`expected_modes_per_cell ≲ 0.65 · probe_dim`. Pushing further
than that adds time without adding modes.

If you don't know the mode count in advance, use
`find_modes_contour_adaptive`. It applies the same rule of thumb
automatically: each cell is split when it either saturates
(returns ≥ `saturation_factor · probe_dim` modes) or collapses
(returns 0 modes at low depth — almost always means the cell is
over capacity, not actually empty). Recursion stops at
`max_depth=6`. On the 303-mode buffon, it recovers every mode at
`probe_dim ∈ {20, 40, 60}` with a 2-3× overhead vs the hand-tuned
`n_k=8` run — the cost of not pre-knowing the answer.

### Batches of similar-density graphs: `tune_contour_parameters`

If you have many graphs of the same topology / density (e.g. a
sweep over RNG seeds for a buffon-style construction), tune once
and reuse:

```python
from netsalt import (
    buffon_planar_graph_or_your_factory,  # your code
    tune_contour_parameters,
    find_modes_contour_subdivided,
)

representative = make_graph(seed=0)
params, info = tune_contour_parameters(representative, bounds=bounds, probe_dim=40)
print(f"{info['discovered_modes']} modes, batch params: {params}")

for seed in seeds:
    g = make_graph(seed)
    modes = find_modes_contour_subdivided(g, bounds=bounds, **params)
```

`tune_contour_parameters` runs `find_modes_contour_adaptive` once,
observes the mode count, and picks `n_k` so that
`expected_modes_per_cell ≤ 0.65 · probe_dim / safety_factor`
(default `safety_factor=1.5` leaves 50% headroom for batch-instance
variation). `bench_tune_then_batch.py` shows the pattern: tuning
takes ~4 s on a representative buffon graph; each subsequent batch
instance runs in ~2 s vs ~4 s for per-call adaptive — 1.8× median
speedup with 100% mode coverage on the batch.

Re-tune only when graph density / topology changes substantially.
If your graph generator can produce occasional outliers (e.g. tiny
graphs from rare seeds), filter them out before passing to the
batch pipeline; the tuned parameters assume similar instance size.

### Does positional accuracy alone require subdivision?

**No.** Across all working configurations, positional error vs the
gold reference is `~1e-11`–`1e-8` regardless of `n_k` once you've
cleared the coverage threshold. Tightening the match tolerance
from `1e-2` to `1e-6` doesn't change which methods agree. On the
22-mode buffon workload, every `n_k ∈ {1, 2, 4, 8, 16}` returns the
same 22 modes at indistinguishable accuracy — see the subdivision
sweep section in `results_search.md`.

The grid path missed one mode on the buffon graph: a near-corner mode
that `peak_local_max` merged into a neighbour at `min_distance=2`.
Beyn caught it.

## Adding a workload

Add an entry to either script's `workloads`/`cases` list and pass a
graph constructed via `_common.line_graph`,
`_common.buffon_planar_graph`, or your own factory. The cross-method
consistency check is built into both scripts — a regression that
makes one method drift from the consensus will fail the run loudly.
