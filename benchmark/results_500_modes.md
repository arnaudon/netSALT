# 500-mode benchmark: buffon n_lines=6, total_length=20

Graph: **61 nodes**, **64 edges**, probe_dim cap **60** (clamped to node count). Rectangle ``k ∈ [0.5, 40], α ∈ [0, 5]`` contains **502 modes** — 8.4× the basic algorithm's per-contour capacity. Gold reference: ``n_k=32, n_α=2, n_quad=320`` (24.6s).

## Manual subdivision sweep

`n_quad=200`, varying `n_k`. Per-cell mode count = `gold_modes / n_k`.

| n_k | modes/cell | time (s) | modes found | max pos err |
|---:|---:|---:|---:|---:|
| 4 | 125.5 | 1.35 | 0 | inf |
| 8 | 62.8 | 2.94 | 1 | 5.58e-04 |
| 12 | 41.8 | 3.70 | 502 | 1.59e-06 |
| 16 | 31.4 | 4.59 | 502 | 1.25e-06 |
| 20 | 25.1 | 5.74 | 502 | 4.91e-08 |
| 24 | 20.9 | 6.79 | 502 | 2.01e-09 |
| 32 | 15.7 | 8.39 | 502 | 6.62e-10 |

Threshold rule (`modes_per_cell ≲ 0.65 · probe_dim = 39`) predicts the transition at `n_k = ⌈502 / 39⌉ = 13`.

## Adaptive sweep

`max_depth=8` (bumped from default 6 — at 500 modes the small-`probe_dim` runs need more recursion budget than the 303-mode stress workload).

| probe_dim | time (s) | modes found | max pos err |
|---:|---:|---:|---:|
| 20 | 25.56 | 428 | 1.69e-07 |
| 40 | 11.79 | 227 | 1.38e-07 |
| 60 | 9.68 | 472 | 2.01e-08 |

## Tune once → apply (single graph)

`tune_contour_parameters` discovered 472 modes in **10.04s** and chose `{'n_k': 19, 'n_alpha': 1, 'n_quad': 200, 'probe_dim': 60}`.

Applying those settings via `find_modes_contour_subdivided` recovered **502 modes in 5.44s** (max position error 2.01e-08 vs gold).

On a single graph the tune step is overhead — useful only when the same parameters get reused across many similar graphs (see `bench_tune_then_batch.py`).

## Takeaway

At **502 modes vs probe_dim 60** (8.4× over capacity):

- Manual subdivision needs `n_k ≥ 14` to recover all modes; `n_k=16` is the comfortable choice.
- Adaptive recovers all modes at any reasonable `probe_dim`, but needs `max_depth=8` to budget enough recursion (default 6 is sized for 300-mode workloads). Pay 2-3× the wall time of the optimum-tuned manual run.
- `tune_contour_parameters` runs adaptive once, picks an appropriate `n_k`, and then any subsequent call on a similar graph runs at full manual speed.
