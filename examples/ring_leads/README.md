# Ring resonator with two leads

A closed loop made open by two pendant lead edges on opposite sides, so the
loop modes can radiate out. Both `linear` and `full_salt_newton` lase the same
two modes; above threshold the newton curves carry the full-SALT saturation
bend.

`bash run.sh` writes `li_curves.png` (geometry | per-mode L–I | total) here.
Figures are not committed; re-run to reproduce. See `doc/source/lasing.rst`
for the solver physics.
