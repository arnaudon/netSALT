# Chord-count sweep — density alone does not buy lasing modes

One 14-node ring with two leads and a growing, nested set of random chords
(6 → 10 → 14; the 6-chord set is `../chaotic_ring`). The naive expectation —
more loops, denser spectrum, more lasing modes — is falsified by the measured
sweep:

| chords | modes in window | mean participation | linear lases | newton lases |
|---|---|---|---|---|
| 6 | 6 | 7.9 | 3 | 4 |
| 10 | 8 | 8.4 | 1 | 1 |
| 14 | 7 | 8.9 | 1 | 1 |

Adding chords delocalises the modes (participation grows) and reshuffles the
cluster under the gain, so overlap rises and the first lasing mode clamps the
gain for the rest — winner-take-all. Multimode lasing needs *localised* modes:
compare `../chaotic_ring` (a localised 4-mode cluster), `../ring_chain`
(localisation by detuning, one mode per ring) and `../mini_buffon` (extended
disorder modes, 2 of 10 lase).

`bash run.sh` writes `chord_sweep.png` here (takes a few minutes). Figures are
not committed; re-run to reproduce.
