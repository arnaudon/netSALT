### buffon n_lines=10 (125 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.33 | 33.4 | 10.7 | 10/10 |
| 10 | brownian | 1.66 | 166.4 | 60.1 | 10/10 |
| 30 | root | 0.96 | 32.1 | 10.8 | 30/30 |
| 30 | brownian | 4.72 | 157.3 | 57.6 | 30/30 |
| 50 | root | 1.61 | 32.1 | 10.5 | 50/50 |
| 50 | brownian | 8.31 | 166.1 | 59.3 | 50/50 |

### buffon n_lines=15 (189 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.35 | 35.2 | 10.3 | 10/10 |
| 10 | brownian | 1.60 | 159.6 | 48.6 | 10/10 |
| 30 | root | 1.02 | 33.9 | 9.8 | 30/30 |
| 30 | brownian | 5.20 | 173.3 | 53.5 | 30/30 |
| 50 | root | 1.73 | 34.5 | 10.0 | 50/50 |
| 50 | brownian | 8.56 | 171.2 | 52.8 | 50/50 |

### buffon n_lines=20 (340 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.55 | 54.9 | 10.8 | 10/10 |
| 10 | brownian | 2.30 | 230.2 | 49.3 | 10/10 |
| 30 | root | 1.54 | 51.5 | 9.7 | 30/30 |
| 30 | brownian | 7.10 | 236.6 | 50.0 | 30/30 |
| 50 | root | 2.49 | 49.8 | 9.4 | 50/50 |
| 50 | brownian | 11.50 | 229.9 | 48.0 | 50/50 |

## Summary: ms/refine at the largest batch per graph size

| nodes | root | brownian | brownian/root |
|---:|---:|---:|---:|
| 125 | 32.1 | 166.1 | 5.2× |
| 189 | 34.5 | 171.2 | 5.0× |
| 340 | 49.8 | 229.9 | 4.6× |

