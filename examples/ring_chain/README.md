# Detuned ring chain — the mode count as a dial

Four rings of increasing size bridged in sequence. Detuning localises one mode
per ring near the gain centre (k = 3.61/3.66/3.73/3.83, 87–100 % on the home
ring); the output leads hang off the **bridge midpoints**, so they selectively
damp the hybridised (delocalised) modes while the one-per-ring quartet keeps
only the uniform material-loss floor. Result: **four co-lasing modes, one per
ring** under both `linear` and `full_salt_newton`, with onsets staggered by
loss and gain detuning (ring 2's mode is the leakiest and turns on last). Add
a ring, add a mode. The mode-profile figure shows each mode on its own ring
with near-zero saturated reshaping — weak competition is *why* they co-lase.

`bash run.sh` writes `ring_chain.png` + `mode_profiles.png` here. Figures are
not committed; re-run to reproduce. See `doc/source/lasing.rst` for the
physics.
