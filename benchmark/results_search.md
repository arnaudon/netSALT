## Does positional accuracy require subdivision?

Short answer: **no, not for accuracy** — only for *mode coverage*.

On a workload where the rectangle's mode count is comfortably below ``probe_dim`` (e.g. buffon, 60 nodes, 22 modes), Beyn returns positions accurate to ``~1e-11``-``1e-13`` from a single contour. Tightening the match tolerance from ``1e-2`` to ``1e-6`` (this run uses ``MATCH_TOL = 1e-06``) doesn't change which methods agree, and at fixed ``n_quad`` / ``probe_dim``, position error vs the gold reference is indistinguishable across ``n_k ∈ {1, 2, 4, 8, 16}``.

Subdivision matters for a different reason: Beyn caps at ``probe_dim`` modes per contour (the SVD step's reduced matrix is ``r × r`` with ``r ≤ probe_dim``). When the rectangle's true mode count *exceeds* ``probe_dim``, the SVD's smallest singular values are below ``svd_tol·σ_max`` and the rank cut drops them — empirically, *all* of them, so the single contour returns 0 modes rather than a truncated set. Subdivision splits the rectangle until each cell has fewer than ``probe_dim`` modes, recovering full coverage. The sweep table below makes this visible: on the line graphs (16 / 21 nodes, 38 / 50 modes in the rectangle), ``n_k=1, 2`` return 0; ``n_k=4`` recovers most or all; ``n_k=8`` is the safe pick. On the buffon graph (60 nodes, 22 modes), every ``n_k`` returns the same set.

### line n=15 (k ∈ [0.5,60], α ∈ [0,5]) — ~38 modes

| method | time (ms) | n_modes | worst `\|λ\|` | max pos err vs gold |
|---|---:|---:|---:|---:|
| contour | 214.8 | 0 | nan | n/a |
| contour-subdiv | 1476.9 | 38 | 1.33e-11 | 4.06e-08 |
| grid+root | 22557.3 | 36 | 1.59e-10 | 4.06e-08 |

### line n=20 (k ∈ [0.5,80], α ∈ [0,5]) — ~50 modes

| method | time (ms) | n_modes | worst `\|λ\|` | max pos err vs gold |
|---|---:|---:|---:|---:|
| contour | 211.9 | 0 | nan | n/a |
| contour-subdiv | 2414.7 | 50 | 9.88e-08 | 1.42e-08 |
| grid+root | 36396.4 | 48 | 3.75e-10 | 3.13e-09 |

### buffon n_lines=6, ~60 nodes (k ∈ [0.5,40], α ∈ [0,5]) — ~22 modes

| method | time (ms) | n_modes | worst `\|λ\|` | max pos err vs gold |
|---|---:|---:|---:|---:|
| contour | 307.5 | 22 | 7.87e-13 | 1.55e-13 |
| contour-subdiv | 1088.1 | 22 | 3.23e-11 | 6.10e-12 |
| grid+root | 27307.4 | 22 | 9.70e-13 | 2.77e-13 |


## Subdivision sweep (n_k vs accuracy / coverage)

### subdivision sweep: line n=15 (k ∈ [0.5,60], α ∈ [0,5]) — ~38 modes

| n_k | time (ms) | n_modes | max pos err vs gold |
|---:|---:|---:|---:|
| 1 | 183.3 | 0 | n/a |
| 2 | 361.7 | 0 | n/a |
| 4 | 836.1 | 38 | 1.74e-07 |
| 8 | 1460.3 | 38 | 4.06e-08 |
| 16 | 2818.8 | 38 | 4.06e-08 |

### subdivision sweep: line n=20 (k ∈ [0.5,80], α ∈ [0,5]) — ~50 modes

| n_k | time (ms) | n_modes | max pos err vs gold |
|---:|---:|---:|---:|
| 1 | 214.1 | 0 | n/a |
| 2 | 473.1 | 0 | n/a |
| 4 | 932.2 | 47 | 5.27e-05 |
| 8 | 1785.9 | 50 | 3.13e-09 |
| 16 | 3620.6 | 50 | 3.13e-09 |

### subdivision sweep: buffon n_lines=6, ~60 nodes (k ∈ [0.5,40], α ∈ [0,5]) — ~22 modes

| n_k | time (ms) | n_modes | max pos err vs gold |
|---:|---:|---:|---:|
| 1 | 292.7 | 22 | 1.55e-13 |
| 2 | 562.6 | 22 | 1.43e-13 |
| 4 | 1095.1 | 22 | 6.10e-12 |
| 8 | 2222.7 | 22 | 1.80e-11 |
| 16 | 4166.3 | 22 | 1.75e-12 |


## Cross-method agreement vs gold

- [line n=15 (k ∈ [0.5,60], α ∈ [0,5]) — ~38 modes] contour vs gold (n_k=8, n_α=2, n_quad=240): matched=0 missed=38 spurious=0
- [line n=15 (k ∈ [0.5,60], α ∈ [0,5]) — ~38 modes] contour-subdiv vs gold (n_k=8, n_α=2, n_quad=240): matched=38 missed=0 spurious=0
- [line n=15 (k ∈ [0.5,60], α ∈ [0,5]) — ~38 modes] grid+root vs gold (n_k=8, n_α=2, n_quad=240): matched=36 missed=2 spurious=0
- [line n=20 (k ∈ [0.5,80], α ∈ [0,5]) — ~50 modes] contour vs gold (n_k=8, n_α=2, n_quad=240): matched=0 missed=50 spurious=0
- [line n=20 (k ∈ [0.5,80], α ∈ [0,5]) — ~50 modes] contour-subdiv vs gold (n_k=8, n_α=2, n_quad=240): matched=50 missed=0 spurious=0
- [line n=20 (k ∈ [0.5,80], α ∈ [0,5]) — ~50 modes] grid+root vs gold (n_k=8, n_α=2, n_quad=240): matched=48 missed=2 spurious=0
- [buffon n_lines=6, ~60 nodes (k ∈ [0.5,40], α ∈ [0,5]) — ~22 modes] contour vs gold (n_k=8, n_α=2, n_quad=240): matched=22 missed=0 spurious=0
- [buffon n_lines=6, ~60 nodes (k ∈ [0.5,40], α ∈ [0,5]) — ~22 modes] contour-subdiv vs gold (n_k=8, n_α=2, n_quad=240): matched=22 missed=0 spurious=0
- [buffon n_lines=6, ~60 nodes (k ∈ [0.5,40], α ∈ [0,5]) — ~22 modes] grid+root vs gold (n_k=8, n_α=2, n_quad=240): matched=22 missed=0 spurious=0
