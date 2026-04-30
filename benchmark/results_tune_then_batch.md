# Tune-once, batch-process: contour parameters across randomized graphs

Representative graph: ``buffon_planar_graph(n_lines=6, total_length=8, seed=2)`` — 150 modes discovered by the adaptive pass in **4.26s**.

Tuned parameters: ``{'n_k': 9, 'n_alpha': 1, 'n_quad': 200, 'probe_dim': 40}``

Batch: 4 additional randomized seeds, same construction.

| seed | nodes | tuned (s) | tuned modes | adaptive (s) | adaptive modes |
|---:|---:|---:|---:|---:|---:|
| 3 | 62 | 2.29 | 150 | 4.25 | 150 |
| 4 | 71 | 2.46 | 150 | 4.54 | 150 |
| 5 | 47 | 2.21 | 150 | 4.07 | 150 |
| 6 | 47 | 2.15 | 151 | 3.88 | 151 |

**Median speedup tuned vs adaptive: 1.8×.** **Median coverage tuned/adaptive: 100%** — the tuned non-adaptive path recovers the same modes as the adaptive path on each batch instance, while skipping the saturation-driven recursion overhead.


**Pattern**: call ``tune_contour_parameters`` once on a representative graph, then splat the returned dict into ``find_modes_contour`` for every other graph in the batch. Re-tune only when graph density / topology changes substantially.

