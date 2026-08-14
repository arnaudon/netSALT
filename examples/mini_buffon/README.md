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
- **This graph exposed the solver's resolution limit**: the Δk = 0.004 pair
  swaps identities at active-set events, and around the **mode crossing** at
  D0 ≈ 0.6 (~20× threshold), where mode 8 overtakes mode 6, the frozen-field
  single-pole iteration cannot resolve the per-mode split — robustly across
  relaxation and step size, so it is a method limit, not a tuning bug.

  Earlier revisions hid this behind a *total-output ratchet* (holding a
  collapsing mode so the summed L–I stayed monotone) and a *wrong-basin guard*
  (reverting an activation that suppressed an established mode). Both have been
  **removed**: they imposed the expected answer on the numerics, which made the
  very thing worth investigating — a non-monotone curve, a mode swap —
  impossible to observe. What replaces them is reporting. Every pump step
  records its SALT residual, active set and convergence into
  `modes_df.attrs["salt_diagnostics"]`, so the crossing region shows up as data
  rather than being smoothed over. Read that table before trusting per-mode
  magnitudes there; the totals are far better determined than the split.

  What was kept from that work, because it is numerics rather than physics:
  linear-slope warm starts (a floor start sits in the trivial `a = 0` basin) and
  a candidate-spacing cap on the k-window (a locality constraint on a local
  solver). Fully resolving the crossing would need a constant-flux-state SALT
  solver — see issue #50.

For many co-lasing modes by *design* (localisation, not pump), see
`../ring_chain`; for why raw spectral density does not help, `../chord_sweep`.

`bash run.sh` writes the three `*_li_curves.png` / `*_mode_profiles.png` pairs
here (~6 min). Figures are not committed; re-run to reproduce.
