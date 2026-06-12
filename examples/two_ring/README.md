# Detuned two-ring "photonic molecule" — genuine multimode lasing

Two rings of different sizes (radii 0.9 / 1.25) joined by a bridge, with a lead
on each. The detuning breaks the left/right symmetry and localises each mode
onto one ring (identical rings would give symmetric/antisymmetric modes spread
over both, with high overlap). Localised modes barely compete, so with a narrow
gain on a cross-ring pair `full_salt_newton` lases several modes spread across
the two rings — genuine multimode lasing, with the modes labelled by ring in
the figure.

`bash run.sh` writes `two_ring_multimode.png` here. Figures are not committed;
re-run to reproduce. See `doc/source/lasing.rst` for the solver physics.
