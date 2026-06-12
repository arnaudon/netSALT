# Mini-buffon network — the dense-spectrum reality check

A shrunk Nat. Commun.-style buffon network (10 random lines, giant component:
39 nodes, 45 edges, 18 radiating lead ends; fixed seed). The spectrum is
genuinely dense — 10 modes in the scan window, ~25 per unit k, matching the
Weyl estimate nL/π ≈ 29 — with disorder-spread losses and a near-degenerate
pair split by Δk = 0.006.

Measured findings (encoded in the script):

- **Competition, not solver cost, limits the lasing count.** Even with the
  gain broadened over all ten modes, only **two** lase under a uniform pump:
  the extended disorder modes overlap strongly and the winners clamp the gain.
  Multimode operation on buffon networks comes from **pump optimisation**
  (`netsalt.pump`), not uniform pumping; for many co-lasing modes by design
  see `../ring_chain`.
- **`full_salt_newton` is still comfortable at this scale**: ~25 s for the
  sweep on the ~700-node oversampled work graph (vs ~0.1 s for `linear`),
  agreeing with `linear` on the lasing set, with the expected
  far-above-threshold deviation in the magnitudes (D0_max ≈ 10× threshold).

`bash run.sh` writes `li_curves.png` + `mode_profiles.png` here (~2 min).
Figures are not committed; re-run to reproduce.
