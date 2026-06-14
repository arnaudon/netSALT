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
  bootstrap capturing the trivial a = 0 root, an add flipping a stronger
  veteran into a weaker newcomer). Fixed in `full_salt_newton` by
  linear-slope warm starts, a candidate-spacing cap on the k-window, and
  continuity guards keyed on physical invariants (lower-threshold veteran
  killed, total-output monotonicity). The coarse and fine grids now agree up
  to a genuine **mode crossing** at D0 ≈ 0.6 (~20× threshold) where mode 8
  overtakes mode 6 — there the frozen-field single-pole iteration cannot
  resolve the per-mode split (it falls to a spurious lower-total branch,
  robustly across relaxation/step size: a method limit). A **total-output
  ratchet** enforces the hard physical law (total cannot fall as pump rises)
  by holding the collapsing mode — the dominant curve plateaus (flagged) and
  the **total L–I stays monotone and physical**. The per-mode magnitudes
  across the crossing are at the method's resolution limit; fully resolving
  it would need a constant-flux-state SALT solver.

For many co-lasing modes by *design* (localisation, not pump), see
`../ring_chain`; for why raw spectral density does not help, `../chord_sweep`.

`bash run.sh` writes the three `*_li_curves.png` / `*_mode_profiles.png` pairs
here (~6 min). Figures are not committed; re-run to reproduce.
