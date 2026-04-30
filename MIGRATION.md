# Migrating from the Luigi workflow to `netsalt.pipeline`

The Luigi task graph (`netsalt/tasks/`) and INI configs (`luigi.cfg`)
have been replaced with a plain-Python pipeline (`netsalt/pipeline.py`)
and YAML configs validated through `NetSaltParams`. This is a clean
break — the Luigi entry points are removed, not deprecated.

## Why

The DAG was a near-linear chain of ~26 tasks, always run with
`--local-scheduler` (Luigi's central scheduler / web UI was never
used). The single value Luigi was actually providing — output-cache
skip semantics — is replaced by a small `if out.exists(): return
load(out)` check at the top of each step function. Configuration moves
to typed YAML so it composes naturally with `NetSaltParams` (which
already validated runtime config).

## Running a workflow

Old:

```bash
luigi --module netsalt.tasks.workflow ComputeLasingModes \
      --local-scheduler --log-level INFO
```

New:

```bash
python -m netsalt lasing config.yaml          # was ComputeLasingModes
python -m netsalt passive config.yaml         # was ComputePassiveModes
python -m netsalt controllability config.yaml # was ComputeControllability + PlotControllability

# Force re-execution even when cached outputs exist:
python -m netsalt lasing config.yaml --force  # was --rerun / --rerun-all
```

The Python API is just as direct:

```python
from netsalt.config_loader import load_config
from netsalt.pipeline import compute_lasing_modes

params = load_config("config.yaml")
compute_lasing_modes(params)
```

## Converting `luigi.cfg` → `config.yaml`

The INI section structure goes away. Everything becomes flat top-level
YAML keys. A few names are renamed for clarity (mostly to remove
ambiguous bare names like `mode` and `method`).

| Old (`section.field`)                         | New (YAML key)                  |
| --------------------------------------------- | ------------------------------- |
| `core.autoload_range`                         | *(drop — Luigi-specific)*       |
| `core.logging_conf_file`                      | *(drop — configure logging in your script if needed)* |
| `CreateQuantumGraph.graph_path`               | `graph_path`                    |
| `CreateQuantumGraph.graph_mode`               | `graph_mode`                    |
| `CreateQuantumGraph.inner_total_length`       | `inner_total_length`            |
| `CreateQuantumGraph.max_extent`               | `max_extent`                    |
| `CreateQuantumGraph.dielectric_mode`          | `dielectric_mode`               |
| `CreateQuantumGraph.method`                   | `dielectric_method`             |
| `CreateQuantumGraph.custom_index`             | `custom_index_path`             |
| `CreateQuantumGraph.inner_value`              | `dielectric_inner_value`        |
| `CreateQuantumGraph.loss`                     | `dielectric_loss`               |
| `CreateQuantumGraph.outer_value`              | `dielectric_outer_value`        |
| `CreateQuantumGraph.node_loss`                | `node_loss`                     |
| `CreateQuantumGraph.edge_size`                | `edge_size`                     |
| `CreateQuantumGraph.k_a`                      | `k_a`                           |
| `CreateQuantumGraph.gamma_perp`               | `gamma_perp`                    |
| `CreateQuantumGraph.keep_degree_two`          | `keep_degree_two`               |
| `CreateQuantumGraph.noise_level`              | `noise_level`                   |
| `ModeSearchConfig.n_workers`                  | `n_workers`                     |
| `ModeSearchConfig.k_n` / `k_min` / `k_max`    | `k_n` / `k_min` / `k_max`       |
| `ModeSearchConfig.alpha_n` / `alpha_min` / `alpha_max` | `alpha_n` / `alpha_min` / `alpha_max` |
| `ModeSearchConfig.quality_threshold`          | `quality_threshold`             |
| `ModeSearchConfig.search_stepsize`            | `search_stepsize`               |
| `ModeSearchConfig.max_steps`                  | `max_steps`                     |
| `ModeSearchConfig.max_tries_reduction`        | `max_tries_reduction`           |
| `ModeSearchConfig.reduction_factor`           | `reduction_factor`              |
| `ModeSearchConfig.quality_method`             | `quality_method`                |
| `ModeSearchConfig.threshold_abs`              | `threshold_abs`                 |
| `ModeSearchConfig.min_distance`               | `min_distance`                  |
| `PumpConfig.D0_max`                           | `D0_max`                        |
| `PumpConfig.D0_steps`                         | `D0_steps`                      |
| `CreatePumpProfile.mode`                      | `pump_mode`                     |
| `CreatePumpProfile.custom_pump_path`          | `pump_custom_path`              |
| `CreatePumpProfile.threshold_target`          | `pump_threshold_target`         |
| `OptimizePump.optimisation_mode`              | `optimize_pump_method`          |
| `OptimizePump.pump_min_frac`                  | `optimize_pump_min_frac`        |
| `OptimizePump.maxiter`                        | `optimize_pump_maxiter`         |
| `OptimizePump.popsize`                        | `optimize_pump_popsize`         |
| `OptimizePump.seed`                           | `optimize_pump_seed`            |
| `OptimizePump.n_seeds`                        | `optimize_pump_n_seeds`         |
| `OptimizePump.disp`                           | `optimize_pump_disp`            |
| `OptimizePump.eps_min` / `eps_max` / `eps_n`  | `optimize_pump_eps_min` / `optimize_pump_eps_max` / `optimize_pump_eps_n` |
| `OptimizePump.cost_diff_min`                  | `optimize_pump_cost_diff_min`   |
| `ComputeModalIntensities.D0_max`              | `intensities_D0_max`            |
| `ComputeLasingModes.lasing_modes_id`          | `lasing_modes_id`               |
| `ComputeControllability.n_top_modes`          | `n_top_modes`                   |
| `PlotPassiveModes.n_modes`                    | `plot_passive_n_modes`          |
| `PlotPassiveModes.edge_size`                  | `plot_passive_edge_size`        |
| `PlotPassiveModes.mode_ids`                   | `plot_passive_mode_ids`         |
| `PlotPassiveModes.ext`                        | `plot_ext`                      |
| `PlotThresholdModes.n_modes`                  | `plot_threshold_n_modes`        |
| `PlotThresholdModes.edge_size`                | `plot_threshold_edge_size`      |
| `PlotThresholdModes.mode_ids`                 | `plot_threshold_mode_ids`       |
| `PlotThresholdModes.ext`                      | `plot_ext`                      |

The `--rerun` / `--rerun-all` flags become `--force` on the CLI (or
`params["force"] = True` programmatically). Output and figure
directories default to `out/` and `figures/`; override with `outdir:` /
`figdir:` in the YAML if you need different paths.

## Config inheritance

YAML configs can compose with a `defaults:` key:

```yaml
# examples/buffon/buffon_missing_pixel/config.yaml
defaults: ../_base.yaml

pump_mode: custom
pump_custom_path: pump.yaml
```

Inheritance is a single-pass shallow merge: the child file overrides
keys from the base. Multi-level chains work (the buffon multimode and
controllability variants chain three deep). Use this to share common
graph-construction and mode-search parameters across related runs.

## Behaviour parity

- The functional test (`tests/test_functional.py`) runs the new
  pipeline and diffs its `out/` against the same `tests/data/run_simple/out/`
  reference HDF5s as the old Luigi version, with the existing
  `atol=1e-5` tolerance.
- Cache semantics match: a step skips work when its output file
  already exists. To bypass, set `params["force"] = True` (or
  `--force` on the CLI), or delete the relevant file from `out/`.
- The `compute_lasing_modes` entry point produces the same plot set as
  the old `ComputeLasingModes` Luigi wrapper (lasing-related plots
  only, *not* the passive-mode plots — those live in
  `compute_passive_modes`, mirroring the original split).
