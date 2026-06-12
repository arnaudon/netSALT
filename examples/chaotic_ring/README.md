# Single ring + random chords — multimode on one small graph

A 14-node ring with 6 hard-coded random chords and two leads: the buffon
mechanism shrunk down. Each chord closes a loop, so the cavity supports many
interfering path lengths — a dense, irregular spectrum of spatially-distinct
modes (see the participation ratios the script prints). They burn their holes
in different places and co-lase: with a narrow gain on a four-mode cluster
`full_salt_newton` lases 4 modes where the clamping-free `linear` model lases
3. The figure shows the geometry, the full-range L–I, and a zoom on the onset
where the modes switch on in turn.

`bash run.sh` writes `chaotic_ring_multimode.png` here. Figures are not
committed; re-run to reproduce. See `doc/source/lasing.rst` for the physics.
