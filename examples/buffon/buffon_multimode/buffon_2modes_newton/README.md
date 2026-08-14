# Buffon network with the operator-level solver

Same pump-optimised two-mode buffon as `../buffon_2modes`, but with
`intensity_method: full_salt_newton` — operator-level full SALT on the **real
buffon network**, not a shrunk stand-in.

This works because `full_salt_newton` bounds its oversampled operator by
`intensity_oversample_node_cap` (default 3000), so the eigensolve stays a few
thousand nodes *regardless of the cavity's physical length*. On the real buffon
it runs in ~12 s at the default cap and ~6 min at the raised cap set here
(`30000`, ≈ λ/4 resolution — the within-edge floor measured on `line_PRA` for
resolving co-lasing modes). Raise `intensity_oversample_node_cap` /
`intensity_oversample_resolution` for more accuracy at more cost; `linear`
remains the cheap first pass for the lasing count.

Run with `bash run.sh` (needs `../../buffon_uniform/out` populated first).
