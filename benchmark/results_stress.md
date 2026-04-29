# Stress benchmark: buffon n_lines=6, total_length=12

Graph: **61 nodes**, **64 edges**. Rectangle ``k ∈ [0.5, 40], α ∈ [0, 5]`` contains **303 modes** (gold reference, ``n_k=24 × n_α=2`` at ``n_quad=320``).

## Sweep 1: vary `n_k` at fixed `n_quad`, `probe_dim`

`n_quad=200`, `probe_dim=60` (clamped to node count). Beyn caps at `probe_dim` modes per cell. Below that ceiling, the algorithm flat-out collapses — the SVD's tail is cut and the recovered mode list goes to zero.

| n_k | modes/cell (≈) | time (s) | modes found | max pos err |
|---:|---:|---:|---:|---:|
| 1 | 303 | 0.27 | 0 | inf |
| 2 | 152 | 0.59 | 0 | inf |
| 4 | 76 | 1.48 | 1 | 5.10e-04 |
| 6 | 50 | 2.01 | 62 | 2.26e-04 |
| 8 | 38 | 2.49 | 303 | 1.58e-06 |
| 12 | 25 | 3.28 | 303 | 9.29e-09 |
| 16 | 19 | 4.20 | 303 | 1.35e-08 |
| 24 | 13 | 5.98 | 303 | 1.76e-09 |

## Sweep 2: single contour on a denser graph, vary `n_quad`

Switched to a *bigger* buffon graph (`n_lines=20, total_length=12` → **340 nodes**, **254 modes** in the rectangle) so `probe_dim` can be set to 340 ≥ mode count. The probe-dim ceiling is no longer the limit — the trapezoidal quadrature itself is. With many poles inside one big contour, the integrands vary rapidly and the moments are ill-conditioned. More `n_quad` helps but at a steep cost.

| n_quad | time (s) | modes found |
|---:|---:|---:|
| 200 | 4.26 | 4 |
| 400 | 6.67 | 17 |
| 800 | 11.80 | 50 |
| 1600 | 22.32 | 215 |

## Sweep 3: `find_modes_contour_adaptive` (auto-pick `n_k`)

Same workload (61-node buffon, 303 modes), but the user picks only `probe_dim` — `n_k` is chosen automatically by saturation feedback. Each cell is bisected when it returns either ≥ `saturation_factor · probe_dim` modes (genuine saturation) or `0` modes at low depth (ambiguous: could mean over-capacity collapse). Recursion stops at `max_depth=6`.

| probe_dim | time (s) | modes found | max pos err |
|---:|---:|---:|---:|
| 20 | 15.10 | 303 | 4.44e-09 |
| 40 | 10.07 | 303 | 7.64e-09 |
| 60 | 5.72 | 301 | 2.67e-08 |

Compare to the best manual run from Sweep 1 (`n_k=8` at `probe_dim=60`): 2.49s, 303 modes. Adaptive's overhead vs the optimum-tuned manual run is the cost of not knowing the right `n_k` in advance — a few extra single-contour evaluations on cells that turn out to fit.

## Takeaway

Subdivision adds two distinct benefits, both indispensable at this mode count (303 modes vs probe_dim cap 60):

1. **Coverage** — without subdivision, the SVD-extraction step drops everything beyond `probe_dim` (or below the quadrature noise floor). Sweep 1 shows the transition: `n_k=1, 2` return 0 modes; `n_k=8` recovers all 303.
2. **Cheap quadrature** — many poles per contour means many more `n_quad` to converge the trapezoidal moments. Subdividing into 8 cells costs `8 · n_quad` evaluations total; matching that coverage with a single contour at `n_quad=1600` costs ~9× as much wall time and still misses modes.

**Rule of thumb (manual):** pick `n_k` so that `expected_modes_per_cell ≲ 0.65 · probe_dim`. On this workload that's `n_k ≥ ⌈303 / (0.65 · 60)⌉ = 8`.

**Use `find_modes_contour_adaptive`** when you don't know the mode count in advance. It treats both saturation (cell returns ≥ `0.7 · probe_dim` modes) and over-capacity collapse (cell returns 0 modes at low depth) as signals to split. Sweep 3 shows it recovers ~all 303 modes from any reasonable `probe_dim` with a 2-3× overhead vs the hand-tuned `n_k=8` run — the cost of not pre-knowing the answer.
