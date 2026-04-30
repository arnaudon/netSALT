### buffon n_lines=10 (125 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.29 | 29.2 | 11.0 | 10/10 |
| 10 | brownian | 1.48 | 148.4 | 60.1 | 10/10 |
| 30 | root | 0.89 | 29.6 | 10.9 | 30/30 |
| 30 | brownian | 4.28 | 142.8 | 57.6 | 30/30 |
| 50 | root | 1.43 | 28.6 | 10.6 | 50/50 |
| 50 | brownian | 7.48 | 149.6 | 59.3 | 50/50 |

### buffon n_lines=15 (189 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.35 | 35.3 | 10.5 | 10/10 |
| 10 | brownian | 1.51 | 151.2 | 48.6 | 10/10 |
| 30 | root | 1.00 | 33.4 | 9.7 | 30/30 |
| 30 | brownian | 4.82 | 160.6 | 53.5 | 30/30 |
| 50 | root | 1.67 | 33.5 | 10.4 | 50/50 |
| 50 | brownian | 7.82 | 156.3 | 52.8 | 50/50 |

### buffon n_lines=20 (340 nodes)

| batch | method | total (s) | ms/refine | evals/refine | converged |
|---:|---|---:|---:|---:|---:|
| 10 | root | 0.46 | 45.8 | 9.7 | 10/10 |
| 10 | brownian | 2.13 | 213.1 | 49.3 | 10/10 |
| 30 | root | 1.37 | 45.7 | 9.7 | 30/30 |
| 30 | brownian | 6.66 | 221.9 | 50.0 | 30/30 |
| 50 | root | 2.28 | 45.6 | 9.6 | 50/50 |
| 50 | brownian | 10.58 | 211.6 | 48.0 | 50/50 |

## Summary: ms/refine at the largest batch per graph size

| nodes | root | brownian | brownian/root |
|---:|---:|---:|---:|
| 125 | 28.6 | 149.6 | 5.2× |
| 189 | 33.5 | 156.3 | 4.7× |
| 340 | 45.6 | 211.6 | 4.6× |

