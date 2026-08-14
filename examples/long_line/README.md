# Long multimode Fabry–Pérot

The open 1D cavity of `../line_fabry_perot`, four times longer: the
longitudinal mode spacing dk = pi/(nL) ≈ 0.52 packs ~12 modes into the gain
window and five of them lase — the textbook multimode FP laser, driven by
spatial hole burning (adjacent longitudinal modes have shifted standing-wave
patterns, so each finds gain the others left). `linear` lases 4;
`full_salt_newton` adds a fifth late mode the frozen-profile model suppresses,
while the two solvers' **total** L–I curves coincide — per-mode redistribution
with the total preserved, the same pattern validated against Ge–Chong–Stone on
`../line_PRA`.

`bash run.sh` writes `li_curves.png` + `mode_profiles.png` here. Figures are
not committed; re-run to reproduce.
