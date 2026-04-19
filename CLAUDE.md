# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`netsalt` simulates network lasers on quantum graphs using the SALT
approximation. It accompanies Saxena et al., *Nat. Commun.* 13, 6573 (2022)
(arxiv:2203.16974). The code is accessible as a Python library or through a
Luigi workflow driven by `luigi.cfg` files (see `examples/`).

## Layout

- `netsalt/` — library code
  - `quantum_graph.py` — build quantum graph from a networkx graph, set lengths,
    pumps, dielectric, Laplacian / weight / incidence matrices, `mode_quality`
  - `modes.py` — mode search driver (`scan_frequencies`, `find_modes`,
    `find_threshold_lasing_modes`, `pump_trajectories`,
    `compute_mode_competition_matrix`, `compute_modal_intensities`). Runs
    `multiprocessing.Pool` over the scan grid.
  - `algorithm.py` — rough mode detection (skimage `peak_local_max`) and
    Brownian-ratchet refinement
  - `physics.py` — dispersion relations, gamma function, dielectric setter
  - `pump.py` — pump optimisation (`scipy.optimize.differential_evolution`,
    `pulp` LP)
  - `io.py` — `pickle` for graphs, `pandas.to_hdf` for modes/qualities
  - `plotting.py`, `utils.py`
  - `tasks/` — Luigi task graph (`workflow.py` exposes `ComputePassiveModes`,
    `ComputeLasingModes`, `ComputeControllability`)
- `tests/test_functional.py` — one functional test that runs
  `ComputeLasingModes` on a line graph and diffs `out/` against
  `tests/data/run_simple/out/` with `dir_content_diff`
- `examples/` — ready-to-run Luigi configs (buffon, ring, wheel, directed,
  line_PRA, transfer)
- `doc/` — Sphinx sources, published at https://arnaudon.github.io/netSALT/

## Running things

- Install: `pip install -e .`
- Full lint + test matrix: `tox` (envs: `lint`, `py38`, `py39`, `docs`)
- Just the test: `pytest tests/` (requires `mock`, `pytest`, `dir-content-diff`)
- Format: `tox -e format` (black, line length 100)
- Run an example workflow: `cd examples/buffon/buffon_uniform && bash run.sh`
  (sets `OMP_NUM_THREADS=1`; netsalt does its own multiprocessing)

## Conventions

- Black, line length 100. `pycodestyle` ignores `W503,E731,W605,E203`.
- Parameters live in `graph.graph["params"]` — a single mutable dict that
  worker classes also mutate (`WorkerModes.set_search_radii`). Be careful when
  editing: pool workers pickle the graph, so per-call mutation does not leak
  across processes, but in-process mutation does.
- `to_complex(mode) == mode[0] - 1j*mode[1]` (note the **minus** sign: alpha is
  stored as `-imag(k)`). Keep this sign convention.
- RNG: compute functions accept an `rng=` kwarg (a
  `numpy.random.Generator`). If omitted, a fresh generator with fresh
  entropy is used. Pass a seeded `np.random.default_rng(seed)` when you
  need reproducibility.

## Important improvements to prioritise

Ranked by impact vs. effort. None of these are required for the code to run;
they are what an "old code" most needs before further work lands on top.

1. **Test coverage is one functional test.** `tests/test_functional.py` runs
   the full Luigi pipeline and byte-diffs HDF5 output. That catches
   regressions but gives no signal on *what* broke. Add unit tests for the
   load-bearing pieces in isolation: `mode_quality`, `to_complex` /
   `from_complex`, `refine_mode_brownian_ratchet` on a toy graph,
   `construct_laplacian` / `construct_weight_matrix` on a 3-edge line,
   `clean_duplicate_modes`, `pump_cost`. Until this exists, any refactor is
   flying blind.

2. **Modernise packaging.** `setup.py` still guards against Python < 2.7 and
   pins `VERSION = "0.2.0"`. Move to `pyproject.toml` (PEP 621), declare
   `requires-python = ">=3.10"`, drop the 2.7 check. `tox.ini` still targets
   `py38`/`py39` (both EOL) and `.github/workflows/run-tox.yml` uses
   `actions/checkout@v2` + `actions/setup-python@v2` (deprecated). Bump CI to
   `@v4` and to Python 3.10–3.12.

3. **`warnings.filterwarnings("ignore")` at module import in `modes.py:27`.**
   This silences *every* warning for *every* consumer of the library. Worse,
   line 28 promotes `np.ComplexWarning` to an error — and `np.ComplexWarning`
   was removed in NumPy 1.25 (it now lives at `numpy.exceptions.ComplexWarning`).
   On a modern NumPy, importing `netsalt.modes` raises `AttributeError`. Fix
   both: scope the filter to the narrowest block that needs it, and use the
   new path (or `warnings.catch_warnings`).

4. ~~**`pickle` for graph I/O.**~~ **Done.** `save_graph` / `load_graph`
   now default to JSON (node-link format) with a custom encoder for numpy
   arrays, complex numbers, `NetSaltParams`, and registered dispersion
   relations. Pickle still works on `.pkl`/`.gpickle` filenames but
   requires an explicit `allow_pickle=True` on load; otherwise the call
   raises, and save emits a `DeprecationWarning`. Luigi defaults and test
   fixtures updated to `.json`.

5. ~~**Global `np.random.seed(42)` calls.**~~ **Done.** All `np.random.seed`
   calls in the package are gone. `laplacian_quality`, `mode_quality`,
   `refine_mode_brownian_ratchet`, `set_dielectric_constant`, and
   `make_buffon_graph` now accept an `rng` kwarg (a
   `numpy.random.Generator`); `WorkerScan` / `WorkerModes` own a per-instance
   `default_rng(seed)` and thread it through. `_verify_lengths` and
   `optimise_pump` use local generators too. The functional-test fixtures
   under `tests/data/run_simple/out/` were regenerated under the new PCG64
   stream.

6. ~~**`graph.graph["params"]` as a shared mutable dict.**~~ **Done.**
   Replaced with a pydantic `NetSaltParams` model (see `netsalt/params.py`)
   that keeps full dict-style access (`params["k_min"]`, `params.get(...)`,
   `"x" in params`) but runs type validation on construction and
   assignment. Unknown keys remain allowed (`extra="allow"`) so callers
   can still stash problem-specific knobs. `update_parameters` validates
   incoming dicts through the model at the graph boundary. The workers
   still mutate `self.params` in place — that's fine because pydantic
   validates each assignment, but a follow-up could remove the mutation.

7. **`raise Exception(...)` in `physics.py`.** `dispersion_relation_linear`,
   `_resistance`, `_dielectric` all raise the bare `Exception` class with
   typo'd messages ("Please correct provide…"). Use `ValueError` (or a
   module-specific exception) and fix the strings — these are user-facing.

8. ~~**`pandas.to_hdf` without `format=` / `mode=`.**~~ **Done.**
   `save_modes` pins `format="fixed", mode="w"`; `save_qualities` pins
   `format="fixed", mode="a"` so it appends rather than silently
   overwriting the `modes` key.

9. **Docs drift.** `tox -e docs` builds Sphinx from `doc/source`, which has
   one `.rst` per module. Several docstrings are stale (`TODO: get rid of
   params`, `force (bool): I forgot` in `update_parameters`). Worth a pass
   once the API stabilises; until then, fix the "I forgot" placeholder.

Lower priority but worth flagging: add type hints on the public API in
`__init__.py`; remove the unused Python-2 `sys` import in `setup.py`; the
luigi config files in `examples/` reference paths that only work from that
directory — a one-line note in each README would save users time.

## Git / branch policy for this repo

Active development branch: `claude/netsalt-documentation-priorities-PVpiY`.
Default branch on GitHub is `master`. CI runs on pushes to `master` and on all
PRs. Do not push directly to `master`.
