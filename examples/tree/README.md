# Binary-tree splitter

A depth-3 binary tree: one input lead, several leaf leads. Lossy enough that a
single mode lases under both `linear` and `full_salt_newton` (winner-take-all
gain clamping).

`bash run.sh` writes `li_curves.png` (geometry | per-mode L–I | total) here.
Figures are not committed; re-run to reproduce. See `doc/source/lasing.rst`
for the solver physics.
