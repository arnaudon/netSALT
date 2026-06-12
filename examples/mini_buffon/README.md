# Mini-buffon network — how to get more modes lasing on a disordered graph

A shrunk Nat. Commun.-style buffon network (10 random lines, giant component:
39 nodes, 45 edges, 18 radiating lead ends; fixed seed). The spectrum is
genuinely dense — 10 modes in the scan window, ~25 per unit k (Weyl estimate
nL/π ≈ 29) — with disorder-spread losses and a near-degenerate pair at
Δk = 0.006.

Three measured stages (all encoded in `run.py`):

| stage | pump | ceiling | linear lases | newton lases |
|---|---|---|---|---|
| low uniform | all 27 inner edges | 0.4 | 2 | 2 |
| high uniform | all 27 inner edges | 1.2 | 5 | 4 |
| shaped | 20 mode-owned edges | 1.2 | 2 | 3 |

- **Gain clamping suppresses, but not absolutely**: the losers' interacting
  thresholds are finite, just 10–40× their bare ones — sweeping the same
  uniform pump 3× further buys back 4–5 co-lasing modes. On this network
  **pump strength, not pump shaping, is the simplest route to more modes.**
- **Pump shaping selects rather than multiplies**: every shaped pump probed
  (greedy low-cross-saturation targets, several target counts/margins) lased
  *fewer* modes than uniform at the same ceiling — removing pump area raises
  all thresholds faster than the decoupling pays back. Shaping is the
  Nat. Commun. lever for choosing *which* mode lases (`netsalt.pump`).
- **Solver cost is not the constraint at this scale**: `full_salt_newton`
  takes ~25–40 s per sweep on the ~700-node oversampled work graph (vs ~0.1 s
  for `linear`) and tracks `linear`'s sets.
- **This graph hardened the solver**: early runs showed spurious per-mode
  kinks (the Δk = 0.004 pair swapping identities at active-set events, the
  bootstrap capturing the trivial a = 0 root, coarse/fine pump grids landing
  on different branches). Fixed in `full_salt_newton` by linear-slope warm
  starts, a candidate-spacing cap on the k-window, and continuity guards
  with bisected sub-stepping; coarse and fine grids now agree everywhere. A
  reproducible, resolution-stable branch exchange remains near D0 ≈ 0.6
  (dominance passes from mode 6 to mode 8) and is flagged by a solver
  warning — possibly genuine bistable switching, but the falling total at
  the exchange means curves beyond a collapse warning deserve care.

For many co-lasing modes by *design* (localisation, not pump), see
`../ring_chain`; for why raw spectral density does not help, `../chord_sweep`.

`bash run.sh` writes the three `*_li_curves.png` / `*_mode_profiles.png` pairs
here (~6 min). Figures are not committed; re-run to reproduce.
