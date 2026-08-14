# Denser chord ring — newton-vs-linear consistency

A 16-node ring with 10 hard-coded chords: a bigger, more strongly-competing
companion to `../chaotic_ring`. Both solvers share the onset slope at
threshold, so they coincide just above it; above threshold they diverge because
`full_salt_newton` re-solves each lasing mode's profile at the operating pump
(the spatial holes, and hence the competition, shift) while `linear` freezes
the profiles. The signature: the dominant mode tracks linear closely, while the
secondary modes are reshuffled — one lights up earlier and stronger, another
later and largely suppressed.

`bash run.sh` writes `dense_ring_compare.png` here (and prints a per-mode
onset/slope/intensity table). Figures are not committed; re-run to reproduce.
