# line_PRA — the Ge–Chong–Stone 1D slab laser

This example is the partially-pumped 1D slab resonator of Ge, Chong & Stone,
*Steady-state ab initio laser theory: generalizations and analytic results*,
[PRA **82**, 063824 (2010)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.063824)
([arXiv:1008.0628](https://arxiv.org/abs/1008.0628)), Figs. 3/5/6:

- cavity of length 1, open on both sides;
- refractive index n = 1.5 on 0 < x < 0.25 and n = 3 on 0.25 < x < 1;
- gain centre k_a = 15, width gamma_perp = 3;
- only the left half (0 < x < 0.5) pumped.

`create_graph.py` builds it as a 10-edge line graph (one n = 1 lead edge at
each end carrying the open boundary, 8 inner edges of length 0.125).

## Run the example

```bash
bash run.sh           # create the graph, then python -m netsalt lasing config.yaml
```

## Reproduce the Fig. 6 validation

The paper's Fig. 6 shows modal intensity vs pump for two lasing modes, both as
the exact SALT (numerical solution of their Eq. 28, open symbols) and as the
single-pole approximation (SPA, solid lines). netsalt's `linear` solver *is*
the SPA, and `full_salt_newton` is an operator-level SALT that should track
the exact symbols (dominant mode suppressed below the SPA once the second mode
turns on, second mode above it).

```bash
python create_graph.py                    # graph.json / index.yaml / pump.yaml
OMP_NUM_THREADS=1 python compare_to_pra_fig6.py
```

This runs the pipeline (cached in `out/`, so re-runs are fast), overlays both
solvers on the digitized Fig. 6 data, writes
`figures/pra_fig6_compare.pdf`, and prints a checkpoint table. Expected
agreement (D0 = 1.258, the figure edge): `linear` vs paper SPA dominant to
< 1 %; `newton` vs paper exact dominant to ~1 %; second mode within ~10 %
(dominated by a +0.02–0.03 offset of its interacting threshold — paper exact
0.892 — inherited from the shared threshold pipeline, not the newton solve).

`data/ge_fig6_digitized.csv` is the digitized reference data. Regenerate it
from the paper itself with

```bash
python digitize_pra_fig6.py    # downloads the arXiv PDF; needs poppler + scipy
```

(axis calibration from the tick labels is good to ~0.3 %; in the single-mode
regime the exact symbols ride on the SPA lines, so only the line is recovered
there and the `spa_*` series carry small bumps where markers merged into it).
