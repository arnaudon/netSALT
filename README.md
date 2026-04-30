[![DOI](https://zenodo.org/badge/171305314.svg)](https://zenodo.org/badge/latestdoi/171305314)

Code simulating quantum graphs. 
================================

Doc: https://arnaudon.github.io/netSALT/

This code simulates network lasers with the SALT approximation and quantum graphs.

This is the code accompanying the following publications:
```
Sensitivity and spectral control of network lasers
Dhruv Saxena, Alexis Arnaudon, Oscar Cipolato, Michele Gaio, Alain Quentel, Sophia Yaliraki, Dario Pisignano, Andrea Camposeo, Mauricio Barahona, Riccardo Sapienza
```
avaialble on arxiv: https://arxiv.org/abs/2203.16974
and Nat. Comm.: https://www.nature.com/articles/s41467-022-34073-3


## Installation

We recommend [uv](https://docs.astral.sh/uv/) for managing the environment.
From a checkout:
```bash
uv venv
uv pip install -e .
```
or from pypi:
```bash
uv pip install netsalt
```

Plain `pip` still works if you prefer:
```bash
pip install -e .
# or
pip install netsalt
```

## Usage

The code is accessible directly, or via a workflow manager (luigi) and related configuration files, see `examples/`.

### Mode search

Modes inside a complex-`k` rectangle can be located three different
ways. The Luigi pipeline picks one via
`params["mode_search_method"]` (default `"contour"`); direct callers
can pick by importing the function they want.

| entry point | when to use | wired into Luigi? |
|---|---|---|
| `find_modes_contour` | the production search; takes an `n_k × n_alpha` cell layout | **yes** — used by `find_passive_modes(method="contour")`, the Luigi default |
| `find_modes_contour_adaptive` | parameter discovery — you don't know the mode count yet | no — direct API only |
| `tune_contour_parameters` | batches of similar-density randomized graphs | no — direct API only |

**`find_modes_contour` is the recommended production entry point.**
`find_modes_contour_adaptive` is intended for exploration: its
saturation-driven recursion can drop a small fraction of modes near
cell edges (1–5% on 300–500-mode workloads in the benchmarks). For
runs where every mode matters, use `find_modes_contour` with an
explicit `n_k`, picked either by hand from the rule
`n_k ≥ ⌈expected_modes / (0.65 · probe_dim)⌉` or by
`tune_contour_parameters`.

#### The Luigi default

`luigi.cfg` has `mode_search_method = contour`, which routes
`FindPassiveModes` (`netsalt/tasks/passive.py`) through
`netsalt.find_passive_modes(method="contour")` →
`find_modes_contour`. Sensible defaults are picked from
`graph.graph["params"]`:

- `n_k` ≈ `round(k_max - k_min)` (one cell per unit `k` — fine for
  the small ranges netsalt typically scans)
- `n_alpha = 2`
- `n_quad = 80`
- `probe_dim` defaults to `min(40, n_nodes)`

The legacy `mode_search_method = grid` path is still available;
it runs `scan_frequencies` + `peak_local_max` + per-mode
`refine_mode_root` and ignores the contour helpers.

#### Using `find_modes_contour_adaptive` directly (with caveat)

When you don't know how many modes are in your rectangle, the
adaptive variant subdivides on demand: each cell is bisected when
it returns either `≥ saturation_factor · probe_dim` modes (genuine
saturation) or `0` modes at low depth (over-capacity collapse,
empirically the dominant failure mode at high mode count). It does
not need a manual `n_k`.

```python
from netsalt import find_modes_contour_adaptive

modes = find_modes_contour_adaptive(
    graph,
    bounds=(k_min, k_max, alpha_min, alpha_max),
    n_quad=200,
    probe_dim=40,         # or None for min(40, n_nodes)
    max_depth=6,          # 8 for >300 modes
    saturation_factor=0.7,
    rng=np.random.default_rng(0),
)
```

**Caveat — coverage is not guaranteed.** On the 500-mode stress
benchmark, adaptive recovers 472–502 modes depending on `probe_dim`
and `max_depth`; the saturation heuristic plus boundary dedup at
deep recursion can drop modes near cell edges. If 100% coverage
matters, use `find_modes_contour` with an explicit `n_k` instead —
preferably one that `tune_contour_parameters` picked for you.

The adaptive search is not currently wired into `FindPassiveModes`;
if you want it inside a Luigi run, call it from a custom
`NetSaltTask` subclass and write the resulting modes to the
`passive_modes_path` output.

#### Batch processing similar graphs (`tune_contour_parameters`)

If you have many graphs of the same topology and density (for
instance a sweep over RNG seeds), `tune_contour_parameters` runs
the adaptive search once on a representative graph, then returns
parameter values sized for the (more reliable) non-adaptive path:

```python
from netsalt import tune_contour_parameters, find_modes_contour

# Step 1: tune once on a representative graph.
params, info = tune_contour_parameters(
    representative_graph,
    bounds=bounds,
    probe_dim=40,
    n_quad=200,
)
# info["discovered_modes"] = 150
# params = {"n_k": 9, "n_alpha": 1, "n_quad": 200, "probe_dim": 40}

# Step 2: batch-process similar graphs — full coverage on each.
for seed in seeds:
    g = make_graph(seed)
    modes = find_modes_contour(g, bounds=bounds, **params)
```

`tune_contour_parameters` picks `n_k` so each cell has at most
`0.65 · probe_dim / safety_factor` modes (default
`safety_factor=1.5` leaves 50% headroom for instance variation).
The `safety_factor` matters: even if adaptive's discovery
under-counts a few modes, the non-adaptive call sized with the
recommended `n_k` recovers them all (`benchmark/bench_500_modes.md`
shows this — adaptive found 472 of 502, but the tuned `n_k=19`
recovered all 502 via `find_modes_contour`).

#### When to choose what

- **Default Luigi run** (`mode_search_method = contour`) is fine
  for most workloads — it picks `n_k ≈ k_range` and the netsalt
  scan rectangles are small enough that this is well-sized.
- **Use `tune_contour_parameters`** when running the same search
  over many randomized similar graphs, or when you want to lock in
  fast and reliable non-adaptive parameters for a production run.
- **Use `find_modes_contour_adaptive` directly** only as a
  one-shot exploration tool on an unfamiliar graph; **do not**
  rely on it for production where every mode matters.

See `benchmark/README.md` and `benchmark/bench_stress.py` for the
empirical study behind `n_k`-vs-`probe_dim` sizing.

### Mode refinement

`refine_mode` exposes two algorithms: `"root"` (default,
MINPACK `hybr`) and `"brownian"` (legacy ratchet). Pick via
`params["refine_method"]`. `root` is 4–5× faster than `brownian`
across every graph size benchmarked; brownian stays available for
reproducibility with pre-2026 results.


