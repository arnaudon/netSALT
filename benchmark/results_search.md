### line n=15 (k ∈ [0.5,15], α ∈ [0,5])

| method | time (ms) | n_modes | worst |λ| |
|---|---:|---:|---:|
| contour | 103.8 | 9 | 2.83e-11 |
| contour-subdiv | 190.5 | 9 | 1.52e-08 |
| grid+root | 3842.9 | 9 | 4.36e-15 |

### line n=20 (k ∈ [0.5,15], α ∈ [0,5])

| method | time (ms) | n_modes | worst |λ| |
|---|---:|---:|---:|
| contour | 110.2 | 9 | 2.36e-11 |
| contour-subdiv | 192.8 | 9 | 2.65e-08 |
| grid+root | 4151.6 | 9 | 1.67e-14 |

### buffon n_lines=6, ~60 nodes (k ∈ [0.5,20], α ∈ [0,5])

| method | time (ms) | n_modes | worst |λ| |
|---|---:|---:|---:|
| contour | 221.7 | 9 | 1.12e-13 |
| contour-subdiv | 430.3 | 9 | 6.52e-14 |
| grid+root | 4817.8 | 8 | 1.35e-14 |


## Cross-method agreement

- [line n=15 (k ∈ [0.5,15], α ∈ [0,5])] contour vs contour-subdiv: matched=9 missed=0 extra=0
- [line n=15 (k ∈ [0.5,15], α ∈ [0,5])] grid+root vs contour-subdiv: matched=9 missed=0 extra=0
- [line n=20 (k ∈ [0.5,15], α ∈ [0,5])] contour vs contour-subdiv: matched=9 missed=0 extra=0
- [line n=20 (k ∈ [0.5,15], α ∈ [0,5])] grid+root vs contour-subdiv: matched=9 missed=0 extra=0
- [buffon n_lines=6, ~60 nodes (k ∈ [0.5,20], α ∈ [0,5])] contour vs contour-subdiv: matched=9 missed=0 extra=0
- [buffon n_lines=6, ~60 nodes (k ∈ [0.5,20], α ∈ [0,5])] grid+root vs contour-subdiv: matched=8 missed=1 extra=0
