# Stress benchmark: buffon n_lines=6, total_length=12

Graph: **61 nodes**, **64 edges**. Rectangle ``k ∈ [0.5, 40], α ∈ [0, 5]`` contains **303 modes** (gold reference, ``n_k=24 × n_α=2`` at ``n_quad=320``).

## Sweep 1: vary `n_k` at fixed `n_quad`, `probe_dim`

`n_quad=200`, `probe_dim=60` (clamped to node count). Beyn caps at `probe_dim` modes per cell. Below that ceiling, the algorithm flat-out collapses — the SVD's tail is cut and the recovered mode list goes to zero.

| n_k | modes/cell (≈) | time (s) | modes found | max pos err |
|---:|---:|---:|---:|---:|
| 1 | 303 | 0.31 | 0 | inf |
| 2 | 152 | 0.62 | 0 | inf |
| 4 | 76 | 1.72 | 1 | 5.10e-04 |
| 6 | 50 | 2.35 | 62 | 2.26e-04 |
| 8 | 38 | 2.85 | 303 | 1.58e-06 |
| 12 | 25 | 3.74 | 303 | 9.29e-09 |
| 16 | 19 | 4.24 | 303 | 1.35e-08 |
| 24 | 13 | 6.01 | 303 | 1.76e-09 |

## Sweep 2: single contour on a denser graph, vary `n_quad`

Switched to a *bigger* buffon graph (`n_lines=20, total_length=12` → **340 nodes**, **254 modes** in the rectangle) so `probe_dim` can be set to 340 ≥ mode count. The probe-dim ceiling is no longer the limit — the trapezoidal quadrature itself is. With many poles inside one big contour, the integrands vary rapidly and the moments are ill-conditioned. More `n_quad` helps but at a steep cost.

| n_quad | time (s) | modes found |
|---:|---:|---:|
| 200 | 3.48 | 4 |
| 400 | 6.12 | 17 |
| 800 | 10.64 | 50 |
| 1600 | 17.11 | 215 |

## Takeaway

Subdivision adds two distinct benefits, both indispensable at this mode count (303 modes vs probe_dim cap 60):

1. **Coverage** — without subdivision, the SVD-extraction step drops everything beyond `probe_dim` (or below the quadrature noise floor). Sweep 1 shows the transition: `n_k=1, 2` return 0 modes; `n_k=8` recovers all 303.
2. **Cheap quadrature** — many poles per contour means many more `n_quad` to converge the trapezoidal moments. Subdividing into 8 cells costs `8 · n_quad` evaluations total; matching that coverage with a single contour at `n_quad=1600` costs ~6× as much wall time and still misses modes.

**Rule of thumb:** pick `n_k` so that `expected_modes_per_cell ≲ 0.65 · probe_dim`. On this workload that's `n_k ≥ ⌈303 / (0.65 · 60)⌉ = 8`.
