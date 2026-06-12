# Open Fabry–Pérot line

A 1D path graph whose two end edges are vacuum leads (the radiative loss that
sets the lasing threshold). With the narrow gain used here, two longitudinal
modes near the gain centre lase under both `linear` and `full_salt_newton`;
above threshold the newton curves bend below the linear ones — the full-SALT
gain saturation the near-threshold model omits.

`bash run.sh` writes `li_curves.png` (geometry | per-mode L–I | total) here.
Figures are not committed; re-run to reproduce. See `doc/source/lasing.rst`
for the solver physics.
