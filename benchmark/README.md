# Benchmarks

Two scripts to spot-check the algorithms shipped in this branch.

- `bench_refine.py` — `refine_mode_root` vs `refine_mode_brownian_ratchet`
  on a buffon planar graph (~25 modes). Reports total wall time,
  ms / refine, `mode_quality` evaluations / refine, and convergence count.
- `bench_contour.py` — `find_modes_contour` (single + subdivided)
  vs `find_modes_contour_adaptive` on two workloads: ~25 modes (easy)
  and ~300 modes (stress, where single-contour collapses and
  subdivision is mandatory).

## Running

From the repo root, with the package installed editable
(`uv pip install -e .`):

```bash
.venv/bin/python benchmark/bench_refine.py
.venv/bin/python benchmark/bench_contour.py
```

Each script prints a small table to stdout. Both run in well under
a minute on a recent laptop.

## Headline numbers

Local-machine, single-threaded, scipy 1.x, Python 3.12.
Relative ratios are the interesting bit; absolute timings will
swing.

### Refinement

`root` is ~5× faster than `brownian` on the test workload, with
~10 evaluations / refine vs the ratchet's ~50–100. Both converge
on every mode. `root` is the default.

### Mode search

| workload | single contour (`n_k=1`) | tuned `n_k` | adaptive |
|---|---|---|---|
| Easy (~25 modes) | works, fastest | works, slight overhead | works |
| Stress (~300 modes) | **collapses to 0 modes** (probe-dim ceiling) | recovers all modes | recovers most modes |

Single contour is fine when the rectangle has comfortably fewer
modes than `probe_dim`; otherwise use `tune_contour_parameters` to
size `n_k` and run `find_modes_contour` with the result.
`find_modes_contour_adaptive` is convenient for one-shot
exploration but can drop a small fraction of modes near cell
boundaries — see its docstring and the project README for the
full discussion.
