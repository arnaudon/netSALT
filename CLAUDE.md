# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`netsalt` simulates network lasers on quantum graphs using the SALT
approximation. It accompanies Saxena et al., *Nat. Commun.* 13, 6573 (2022)
(arxiv:2203.16974). The code is accessible as a Python library or through
a plain-Python pipeline (`netsalt.pipeline`) driven by YAML config files
(see `examples/`). Earlier releases shipped a Luigi task graph; that was
removed in favour of the pipeline — see `MIGRATION.md` for the field
mapping.

## Layout

- `netsalt/` — library code
  - `quantum_graph.py` — build quantum graph from a networkx graph, set lengths,
    pumps, dielectric, Laplacian / weight / incidence matrices, `mode_quality`
  - `modes.py` — mode search driver (`scan_frequencies`, `find_modes`,
    `find_threshold_lasing_modes`, `pump_trajectories`,
    `compute_mode_competition_matrix`, `compute_modal_intensities`). Runs
    `multiprocessing.Pool` over the scan grid.
  - `algorithm.py` — rough mode detection (skimage `peak_local_max`) and
    two refinement algorithms: `refine_mode_root` (MINPACK ``hybr``,
    default) and `refine_mode_brownian_ratchet` (legacy random-walk
    ratchet). The dispatcher ``refine_mode(...)`` picks one based on
    ``params["refine_method"]``. Newton (Hellmann-Feynman) and
    Nelder-Mead used to live here but were dropped: see
    `benchmark/bench_refine.py` for the wall-time numbers
    that didn't justify the maintenance cost.
  - `contour.py` — Beyn's contour-integration mode search. Locates every
    root of ``det(L(k)) = 0`` inside a complex contour in ``O(N_quad·L²)``
    work. ``find_modes_contour`` is the production entry point — runs
    Beyn on an ``n_k × n_alpha`` grid of sub-contours and dedups at
    cell boundaries. On a buffon graph this is ~40× faster than the
    80-worker grid scan and returns modes at ``|λ₁| ≈ 10⁻¹⁰`` with no
    refinement step. ``find_modes_contour_adaptive`` does
    saturation-driven recursion when the mode count is unknown
    (parameter discovery only — coverage is not guaranteed at deep
    recursion). ``tune_contour_parameters`` is the bridge: run
    adaptive once, get an ``n_k`` sized for the production
    ``find_modes_contour`` call.
  - `physics.py` — dispersion relations, gamma function, dielectric setter
  - `pump.py` — pump optimisation (`scipy.optimize.differential_evolution`,
    `pulp` LP)
  - `io.py` — `pickle` for graphs, `pandas.to_hdf` for modes/qualities
  - `plotting.py`, `utils.py`
  - `pipeline.py` — plain-Python pipeline. `step_*` functions are the
    individual cached compute / plot steps; `compute_passive_modes`,
    `compute_lasing_modes`, `compute_controllability` are the entry
    points (replacing the Luigi `Compute*` wrappers). Each step caches
    on disk: it skips when the output file exists unless
    `params["force"]` is truthy.
  - `config_loader.py` — YAML loader with a `defaults: <relative-path>`
    inheritance key. Returns a validated `NetSaltParams`.
  - `__main__.py` — CLI dispatcher. `python -m netsalt
    {passive|lasing|controllability} <config.yaml> [--force]`.
- `tests/test_functional.py` — one functional test that runs
  `compute_lasing_modes` on a line graph and diffs `out/` against
  `tests/data/run_simple/out/` with `dir_content_diff`
- `examples/` — ready-to-run YAML configs (buffon, ring, wheel, directed,
  line_PRA, transfer). Buffon variants use `defaults:` to inherit shared
  base configs.
- `doc/` — Sphinx sources, published at https://arnaudon.github.io/netSALT/

## Running things

- Install: `uv pip install -e .` (or `pip install -e .`). `uv venv` to make a
  fresh environment first.
- Full lint + test matrix: `tox` (envs: `lint`, `py310`, `py311`, `py312`,
  `coverage`, `docs`). `tox-uv` is picked up automatically when present.
- Just the test: `pytest tests/` (requires `mock`, `pytest`, `dir-content-diff`)
- Format: `tox -e format` (ruff, line length 100)
- Run an example workflow: `cd examples/buffon/buffon_uniform && bash run.sh`
  (sets `OMP_NUM_THREADS=1`; netsalt does its own multiprocessing). The
  underlying invocation is `python -m netsalt lasing config.yaml`.

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
   the full pipeline and byte-diffs HDF5 output. That catches
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

10. ~~**Luigi workflow + INI configs.**~~ **Done.** `netsalt/tasks/` is
    deleted; orchestration is now `netsalt/pipeline.py` (cached step
    functions) plus `netsalt/__main__.py` (CLI dispatcher). Configs are
    YAML loaded into `NetSaltParams` via `netsalt/config_loader.py`,
    with `defaults: <relative-path>` for inheritance. Luigi is removed
    from `pyproject.toml`. Migration mapping is in `MIGRATION.md`.

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

9. ~~**Docs drift.**~~ **Done.** `tox -e docs` builds clean, the stray
   `[paper]` citation is replaced with the published Nat. Commun.
   reference, and the stale ``I forgot`` / ``TODO`` placeholders in
   ``quantum_graph.py`` were updated. ``netsalt/params.py`` has its own
   `params.rst` in the autodoc tree.

## Lower-priority items landed

- Example scripts and the functional-test fixture all use ``.json``
  for graphs and ``.yaml`` for configs. The lone checked-in
  ``buffon.gpickle`` was regenerated as ``buffon.json``. The
  controllability flow writes ``single_mode_matrix.npy`` (via
  ``np.save``) instead of pickle; the optimised pump results file is
  now ``.npz``.
- Dead ``# pylint: disable=…`` comments removed (pylint isn't in the
  toolchain anymore).
- ``B905`` / ``B007`` lint rules are on. All ``zip(...)`` call sites
  that pair equal-length sequences carry ``strict=True``.
- Type hints added to the simpler public helpers (``utils.py``,
  ``physics.gamma``, ``io.py`` signatures). ``__init__.py`` now has an
  explicit ``__all__``.
- New ``TestComputeCore`` unit tests smoke-cover ``construct_laplacian``
  / ``construct_weight_matrix`` / ``construct_incidence_matrix`` on a
  fresh line graph, and assert ``mode_quality`` is deterministic given
  the same ``rng`` seed.

## Known design debt (follow-up PRs)

- ``WorkerModes`` still mutates ``graph.graph["params"]`` in place to
  stash the current ``D0`` and search window. Downstream consumers
  (``mode_on_nodes``, ``pump_linear``, the dispersion relations) read
  those fields back off the graph to reconstruct the laplacian at the
  right ``D0``, so decoupling requires carrying ``(mode, D0)`` pairs
  explicitly through the modes dataframe — tracked as architectural
  debt rather than a quick fix.
- Compute-core tests are still fairly shallow. Adding a test for
  ``compute_mode_competition_matrix`` and ``find_threshold_lasing_modes``
  on a tiny analytic graph would give more regression coverage.

## Git / branch policy for this repo

Active development branch: `claude/netsalt-documentation-priorities-PVpiY`.
Default branch on GitHub is `master`. CI runs on pushes to `master` and on all
PRs. Do not push directly to `master`.
