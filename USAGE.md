# Usage Guide

End-to-end guide for running experiments, working with real data, writing new
algorithms, and extending the framework.

---

## Table of Contents

1. [Installation](#installation)
2. [Running Experiments](#running-experiments)
3. [Experiment Config Reference](#experiment-config-reference)
4. [Diagnostics](#diagnostics)
5. [Plotting](#plotting)
6. [Real Data Pipeline](#real-data-pipeline)
7. [Sweeps](#sweeps)
8. [Extending the Framework](#extending-the-framework)

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package installs one `bcd-lab` command. Explicit configuration paths work
from any directory; name shortcuts and `list` inspect `configs/` under the
current project directory.

---

## Running Experiments

The package has one command-line interface. Running `bcd-lab` without a
subcommand prints its help. `<config>` may be a path or its canonical key below
`configs/`, such as `examples/quickstart`.

```bash
# Run an experiment by name
bcd-lab run examples/quickstart

# Run and plot immediately
bcd-lab run examples/quickstart --plot

# Plot results from an earlier run
bcd-lab plot examples/quickstart

# List configs under configs/
bcd-lab list
```

By default, results are written to `results/<experiment>/` as JSON files and
plots to `plots/<experiment>/`.
The optional `output_dir` is a results root; the runner creates the experiment
subdirectory beneath it. The optional `results_dir` for `plot` points directly
to a directory containing result JSON files.

Each result JSON uses one canonical structure: `experiment_spec` contains the
complete resolved one-run configuration, including `data.shape`; `run` records
trial context, effective seeds, timing, and termination; and diagnostic records
identify their step with `iteration`.
`summary.metrics` contains purposeful per-metric aggregates for summary plots,
while time-series plots read `diagnostics` directly.

---

## Experiment Config Reference

Configs live in purpose-specific directories under `configs/` and use schema
version 2. `examples/` contains approachable runs, `figures/` contains the four
publication-aligned synthetic studies, `real_data/` contains one primary
MovieLens and one primary Jester run, and `studies/` contains optimizer or
tuning investigations. See `configs/README.md` for status and workload. Start with:

```bash
bcd-lab run examples/quickstart --plot
```

### Minimal example

```yaml
schema_version: 2
experiment: my_run
algorithm:
  name: bcd
  params:
    regularization:
      alpha: 1.0e-3
      lambda_: 0.0
      mu: 1.0
    z_optimizer:
      name: gradient_descent
      params:
        learning_rate: 1.0e-5
    phi_update:
      learning_rate: 1.0e-5
trials: 1

seed: 0

data:
  source: synthetic
  type: standard_normal
  shape:
    n_rows: 100
    n_cols: 100
    rank: 3
    n_observations: 5000

link:
  kind: piecewise_linear_rkhs
  sigma: 1.0
  params:
    grid:
      count: 1000
      bounds:
        name: fixed
        params: {lower: -3.0, upper: 3.0}
      rebuild:
        name: never
        params: {}

objective:
  name: squared_error
  params: {}

runtime:
  max_iter: 100000
  tol: 1.0e-6
  diagnostics_interval: 50
```

`squared_error` is the standard built-in objective. `half_squared` is a
distinct objective with a one-half data-fit scale; it is retained as a compact
example of how objective choice changes optimization and regularization balance.

### Top-level keys

| Key | Description |
|-----|-------------|
| `schema_version` | Must be `2` |
| `experiment` | Name used for results/plots directory |
| `algorithm` | Registered name plus algorithm-owned `params` |
| `trials` | Number of independent runs |
| `seed` | One integer used to derive stable, named random streams per trial |
| `data` | Data source and shape |
| `link` | Link function kind and params |
| `objective` | Registered data-fit objective name and its parameters |
| `runtime` | Iteration limits, tolerance, dtype |
| `diagnostics` | Which metric families and metrics to compute |
| `selection` | Optional observed metric and direction used to retain one coherent model |
| `sweep` | Axes to grid-search over |
| `plot` | Plot definitions |

Relative paths inside a configuration, including `data.path` and `plot.dir`,
are resolved against the YAML file's directory. The experiment `seed` is the
single source of randomness: each trial derives stable, named streams for data,
initialization, and the algorithm without mutating Python or NumPy global state.
When calling `run_sweep` directly, a relative output root follows the same
config-relative rule; omitting it writes under `results/` beside the YAML file.
Run filenames combine a readable prefix with canonical content identity.
Existing planned JSON or checkpoint files are rejected before execution; pass
`--overwrite` to `bcd-lab run`, or `overwrite=True` to the Python API, only
when replacement is intentional.

Shared configuration is validated before data loading or output creation.
Trials and data dimensions must be positive; rank cannot exceed either matrix
dimension; and observations cannot exceed the number of matrix entries.
`runtime.max_iter` and `runtime.tol` are non-negative,
`diagnostics_interval` is null or positive, checkpoint iterations are unique
values within the run, and runtime dtype is `float32` or `float64`. Registered
components remain responsible for validating their own `params`.
Packaged defaults merge only while their corresponding component remains
selected. Changing `data.source`, `link.kind`, or `initialization.mode`
replaces that component block, so parameters from the previous implementation
cannot leak into the new one. Diagnostic `families` are also an explicit
selection: when provided, omitted families are disabled.

### `data` section

**Synthetic:**
```yaml
data:
  source: synthetic
  type: standard_normal      # or: discrete
  shape:
    n_rows: 100
    n_cols: 100
    rank: 3
    n_observations: 5000
  noise_std: 0.01            # optional
  desirable: false           # if true, uses balanced incoherent factorization
  generating_link:
    name: sigmoid             # identity, sigmoid, exponential, tanh, or registered name
    params:
      shifted: true
  evaluation_domain:
    central_mass: 0.95        # retain the 2.5% through 97.5% latent quantiles
    grid_count: 1000
```

`data.generating_link` creates the ground-truth function used to transform the
clean latent matrix before observation noise is added. It is independent from
the top-level `link`, which is the function model fitted by the algorithm. The
synthetic loader publishes both the truth function and the dense transformed
matrix for diagnostics. Omitting the section uses the semantic identity
generator.

`evaluation_domain` is optional and synthetic-loader-owned. When present, its
strictly increasing grid spans the configured central probability mass of the
noise-free latent matrix. `phi` and `phi_sharp` function distances are measured
on that grid, and linked-truth matrix errors include only entries whose latent
truth lies inside its endpoints. Without it, each function representation uses
its documented native distance policy.

The paper's random nonlinear truth is available as:

```yaml
data:
  generating_link:
    name: random_monotone_rkhs
    params:
      degree: 4
      sigma: 1.0
      fit_ridge: 1.0e-6
      num_centers: null       # null uses every sampled latent value
```

The builder draws a strictly increasing polynomial on `xmin` to `xmax`, fits
its sampled values in a Gaussian RKHS, and verifies that the fitted function is
strictly increasing over the center domain. Its parameters also expose
`polynomial_grid_count`, `waviness`, `min_margin`, `min_range`, and
`max_attempts`. Generator construction and observation noise use independent,
named random streams.

**Real data (CSV):**
```yaml
data:
  source: csv
  path: ../data/ml-100k.csv  # relative to this experiment YAML file
  shape:
    rank: 10
  size:
    max_rows: 500
    max_cols: 500
    strategy: top            # or: random
  split:
    validation_fraction: 0.1
    stratify_by: row
```

`n_rows` and `n_cols` are inferred from the loaded data — do not set them manually for CSV sources.

### `link` section

| `kind` | Description |
|--------|-------------|
| `piecewise_linear_rkhs` | Model-owned fixed or input-quantile grid; fast, no kernel solves |
| `gaussian_rkhs` | Gaussian RKHS expansion; flexible, slower |
| `identity` | No link function (linear model) |
| `sigmoid` | Fixed sigmoid |
| `exponential` | Fixed exponential |

```yaml
link:
  kind: piecewise_linear_rkhs
  sigma: 1.0
  params:
    grid:
      count: 50
      bounds:
        name: input_quantiles
        params:
          lower_quantile: 0.01
          upper_quantile: 0.99
          pad_fraction: 0.25
          min_padding: 0.5
      rebuild:
        name: never
        params: {}
```

Grid bounds and rebuilding are independent choices. `fixed` bounds use an
explicit interval and require `rebuild.name: never`. `input_quantiles` derives
the interval from the current factor inner products. Its rebuild policy is
`never` (resolve once during initialization), `every_iteration` (recompute at
each phi-gradient update), or `adaptive` (recompute only when inputs leave the
grid, resolution becomes poor, or estimated function drift exceeds the
configured tolerance). The archived MovieLens/Jester policy is the example
above: 1%/99% quantiles, 25% padding, minimum padding 0.5, fixed after
initialization. Resolved grids and input bounds are checkpointed exactly.

```yaml
link:
  kind: gaussian_rkhs
  sigma: 1.0
  params:
    unconstrained_ridge: 1.0e-8
    monotone_phi_enabled: true
    monotone_phi_direction: increasing
```

---

## Diagnostics

The `diagnostics` section selects which metrics are computed and printed at
each `diagnostics_interval`.

```yaml
diagnostics:
  on_error: raise
  families:
    objective:   [loss]
    timing:      [elapsed]
    gradients:   [z_grad_norm, phi_grad_norm]
    truth_matrix: [x_error, phi_x_error]
    truth_z:     [goodness]
    phi:         [phi_diff, phi_gaussian_rkhs_error, phi_gaussian_rkhs_norm]
    phi_sharp:   [lyapunov, regret, phi_sharp_diff_star, phi_sharp_diff]
```

| Family | Available metrics |
|--------|-------------------|
| `objective` | `loss` |
| `timing` | `elapsed`, `sec_per_iter`, `rem_sec` |
| `gradients` | `z_grad_norm`, `phi_grad_norm` |
| `truth_matrix` | `x_error`, `relative_x_fro`, `x_rmse`, `phi_x_error`, `relative_phi_x_fro`, `phi_x_rmse` |
| `truth_z` | `goodness`, `relative_goodness`, `procrustes_det` |
| `phi` | `phi_diff`, `phi_diff_0`, `phi_sample_mse`, `phi_d_in_star_bounds`, `phi_gaussian_rkhs_error`, `phi_gaussian_rkhs_norm` |
| `phi_sharp` | `lyapunov`, `regret`, `phi_sharp_diff_star`, `phi_sharp_diff` |
| `reconstruction` | `train_rmse`, `train_mae`, `test_rmse`, `test_mae`, `test_relative_rmse` |

Omitting a family disables it entirely. Setting a family to `null` enables all
its metrics. `on_error: raise` is the strict default: unexpected defects stop
the run with diagnostic-family context and retain the original exception as
their cause. Use `on_error: record` only for intentionally best-effort
evaluation; defects are then recorded in `metadata.diagnostic_events` with
their family, metric names, exception type, and message. Missing optional
inputs are not defects: their metric value is `null` and the same metadata list
records the metric as unavailable.

Diagnostics are printed to the terminal and saved in the result JSON under the
`diagnostics` key.

### Validation-selected model

An experiment can retain the complete algorithm state from its best observed
diagnostic record:

```yaml
selection:
  metric: validation_rmse
  mode: min
  retain_checkpoint: true
```

Selection observes the same records as diagnostics: iteration zero, powers of
ten, each configured `runtime.diagnostics_interval`, the final iteration, and
an otherwise-unobserved convergence iteration. A smaller value wins in `min`
mode and a larger value wins in `max` mode; exact ties keep the earlier model.
The metric must be requested and must be finite at every observation. A missing
or non-finite selection value stops the run even when diagnostic failures are
otherwise configured with `on_error: record`.

The result's `selection` object records the chosen iteration and value, the
entire diagnostic record from that iteration, and algorithm-owned JSON exports.
With `retain_checkpoint: true`, the matching Z, phi, and optimizer state is
written to `<run_name>_selected_checkpoint.npz`. This selected archive is
independent of scheduled checkpoint saving, so `runtime.save_checkpoints` may
remain false. Selection is within each run; comparing sweep cases or trials is
a separate analysis step.

On an explicit evaluation grid \(G\), function distance is the numerical
\(L^2\) quantity
\(\sqrt{\sum_{x\in G}(\phi_a(x)-\phi_b(x))^2\,\Delta x}\). Its units are link
output units times the square root of latent-input units. `phi_sample_mse` is
the mean squared link-output error on the same grid. For a piecewise-linear
fit with Gaussian-RKHS truth, `phi_gaussian_rkhs_error` first interpolates the
fit in the truth function's Gaussian kernel span and reports the RKHS norm of
the difference; `phi_gaussian_rkhs_norm` reports the projected fit's RKHS norm.
Because that projection solves on the truth model's complete Gaussian basis,
request these two metrics only at diagnostic intervals appropriate for the
basis size.

`lyapunov` is the combined quantity used by the published analysis:

\[
\Delta_t = \operatorname{goodness}(Z_t,Z^*)
          + \frac{\lVert\phi_t-\phi_t^\sharp\rVert_{L^2}}{\alpha}.
\]

It is available only when factor truth, a positive `alpha`, and
`phi_sharp` are available. It is the sole canonical combined link/factor
metric; the repository does not expose the earlier experimental gap formula
or a legacy metric alias.

---

## Plotting

Plots are defined inline in the config under the `plot` key:

```yaml
plot:
  enabled: true
  dir: ../plots/my_run       # relative to this experiment YAML file
  name: my_run
  max_plots_per_row: 3
  show_legend: false
  plot1:
    type: line
    x: iteration
    y: loss
    x_scale: linear
    y_scale: log
    title: "Loss vs Iteration"
    xlabel: Iteration
    ylabel: Loss
    overlay_sweep: true       # overlay all sweep variants on one axis
  plot2:
    type: selected_link       # reads the validation-selected link export
    show_identity: true
    title: "Selected learned link"
    xlabel: Latent inner product
    ylabel: Prediction
```

- Each `plot<N>` defines one panel.
- `overlay_sweep: true` draws all sweep variants on the same axes.
- `x_scale` / `y_scale`: `linear` or `log`.
- Matplotlib style is cached in `.mplconfig/` to avoid re-initialising fonts on
  every run.

Plot families are registry-driven. Built-in renderers live independently under
`bcd_lab.plotting.renderers`; shared record, transformation, layout, legend,
and output services are separate from figure drawing. Each registered
`PlotRendererDefinition` owns its YAML option normalizer plus its overlay,
legend, and raster capabilities. Custom renderer-specific fields therefore
follow the same public YAML-to-output path as built-ins.

To regenerate plots from existing results without re-running:

```bash
bcd-lab plot examples/quickstart
```

---

## Real Data Pipeline

Raw downloaded files live in `data-raw/` (gitignored).  
Clean triple CSVs live in `data/` (gitignored).  
The experiment runner reads only from `data/`.

### Step 1 — Download raw data

Place raw files under `data-raw/<dataset-name>/`. For MovieLens:

```
data-raw/
  ml-100k/
    u.data
  ml-1m/
    ratings.dat
  ml-latest-small/
    ratings.csv
  jester/
    jester-data-1.xls
```

Download links:
- [ml-100k](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
- [ml-1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip)
- [ml-10m](https://files.grouplens.org/datasets/movielens/ml-10m.zip)
- [ml-20m](https://files.grouplens.org/datasets/movielens/ml-20m.zip)
- [MovieLens latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- [Jester Dataset 1](https://eigentaste.berkeley.edu/dataset/) — download part 1,
  extract its `.xls` file, and place it at the path shown above

The exact paper-data downloads are prepared as follows:

```bash
mkdir -p data-raw
curl -L https://files.grouplens.org/datasets/movielens/ml-latest-small.zip \
  -o data-raw/ml-latest-small.zip
unzip data-raw/ml-latest-small.zip -d data-raw

# Download Jester Dataset 1 part 1 from the official page, then:
mkdir -p data-raw/jester
mv /path/to/jester-data-1.xls data-raw/jester/jester-data-1.xls
```

MovieLens latest-small is pinned to the September 2018 release: 610 users,
9,724 movies, and 100,836 ratings. The Jester converter is pinned to Dataset 1
part 1: 24,983 users, 100 jokes, and 1,810,455 observed ratings. It interprets
the first spreadsheet column as the per-user observation count, removes the
`99` missing-value sentinel, and rejects observed ratings outside `[-10, 10]`.

### Step 2 — Preprocess

```bash
# List available preprocessors
bcd-lab preprocess --list

# Convert ml-100k to data/ml-100k.csv
bcd-lab preprocess ml-100k

# Exact paper-data conversions
bcd-lab preprocess movielens-latest-small
bcd-lab preprocess jester-dataset-1

# Custom directories
bcd-lab preprocess ml-1m --raw-dir /data/downloads --out-dir ./data
```

Output uses canonical `row,col,value` triples. The exact preprocessors also
write `<dataset>.csv.provenance.json`, containing their versioned parameters,
raw source IDs and SHA-256 fingerprints, output fingerprint, identifier-set
digests, dimensions, observation counts, and missing-value counts. MovieLens
source user/movie IDs and Jester's zero-based source row plus one-based joke ID
are retained; the CSV loader creates dense internal mappings per run.

When a canonical CSV has a provenance sidecar, the CSV loader verifies the
output fingerprint before use. The complete preprocessing record is then copied
to `provenance.preprocessing` in every result artifact.

### Step 3 — Reference in config

```yaml
data:
  source: csv
  path: ../data/ml-100k.csv  # relative to this experiment YAML file
  shape:
    rank: 10
```

If the file is missing, the loader identifies the resolved path. Prepare the
tracked MovieLens example with `bcd-lab preprocess ml-100k` as shown above.
That example uses a monotone 50-point initial-quantile piecewise-linear link and a
40-iteration horizon selected from repeated validation-convergence sweeps. Its
factor row-norm projection is active with the selected `mu: 0.6`.

### Adding a new dataset

1. Create `src/bcd_lab/data/preprocessors/<dataset>.py`
2. Subclass `DataPreprocessor`, set `name` and `description`, implement `preprocess()`
3. Call `self.write_preprocessed_triples(...)` with the raw source paths,
   parameters, counts, and compact identifier summaries
4. Import the class and register one instance in `preprocessors/__init__.py`

```python
from bcd_lab.data.provenance import summarize_identifiers

from .base import DataPreprocessor


class MyDataset(DataPreprocessor):
    name = "my-dataset"
    description = "My custom dataset"

    def preprocess(self, raw_dir, out_dir, **kwargs):
        source = raw_dir / "my-dataset" / "ratings.csv"
        # read source and build rows, cols, values lists
        return self.write_preprocessed_triples(
            out_dir,
            self.name,
            rows,
            cols,
            values,
            parameters={"input_format": "ratings.csv"},
            source_paths=[("my-dataset/ratings.csv", source)],
            counts={"output_observations": len(values)},
            identifiers={
                "rows": summarize_identifiers(rows, source_field="user_id"),
                "columns": summarize_identifiers(cols, source_field="item_id"),
            },
        )
```

Built-ins are registered explicitly, in order, by the package `__init__.py`.
External applications may instead call the same public `register_preprocessor`
API during startup. A name can be registered only once; duplicates raise a
clear error and never replace the existing component.

---

## Sweeps

Grid-search over existing run-defining component and configuration fields using
`sweep.axes`:

```yaml
sweep:
  axes:
    - path: algorithm.params.phi_update.learning_rate
      values: [1.0e-4, 1.0e-5, 1.0e-6]
    - path: algorithm.params.regularization.lambda_
      values: [0.0, 0.01, 0.1]
    - path: data.noise_std
      values: [0.0, 0.1]
    - path: initialization.mode
      values: [joint, separate]
    - path: objective.name
      values: [squared_error, half_squared]
```

- `path` is a dot-separated path in the complete experiment specification
  (for example, `data.noise_std`, `diagnostics.families.objective`, or
  `algorithm.params.project_z`)
- All combinations are run as separate trials
- Results for each combination are written to separate JSON files
- Paths must already exist in the base configuration; misspellings fail before
  output is created or work begins
- `seed.*` and `trials` may be swept. Orchestration fields
  (`schema_version`, `experiment`, and `sweep.*`) are reserved and rejected.

When sweeping a component name, every selected implementation must be able to
consume the parameters declared in that case. For components with different
required fields—such as synthetic data and CSV data—use separate experiment
files rather than changing only `data.source`.

Algorithm-owned settings always live under the component's `params` mapping:

```yaml
algorithm:
  name: my_algo
  params:
    learning_rate: 0.01
```

---

## Extending the Framework

### Add an algorithm

Implement the `Algorithm` protocol and register an instance. The runner passes
all inputs by keyword and expects an `AlgorithmResult`:

```python
from typing import Any, Mapping

import numpy as np

from bcd_lab.algorithms import register_algorithm
from bcd_lab.diagnostics import DiagnosticRequest
from bcd_lab.function_models import ScalarFunction
from bcd_lab.losses import Objective
from bcd_lab.types import (
    AlgorithmResult,
    DatasetBundle,
    RunTermination,
    SelectionPolicy,
)


class MyAlgorithm:
    name = "my_algo"

    def build_config(
        self,
        *,
        data: Mapping[str, Any],
        link: Mapping[str, Any],
        runtime: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Validate and return only the configuration this algorithm needs.
        return {"params": dict(params), "runtime": dict(runtime)}

    def run(
        self,
        *,
        config: dict[str, Any],
        dataset: DatasetBundle,
        objective: Objective,
        z_init: np.ndarray | None = None,
        phi_init: ScalarFunction | None = None,
        diagnostics_request: DiagnosticRequest | None = None,
        selection_policy: SelectionPolicy | None = None,
        rng: np.random.Generator,
    ) -> AlgorithmResult:
        _ = selection_policy
        z = z_init
        if z is None:
            z = np.zeros((dataset.n_rows + dataset.n_cols, dataset.rank))
        return AlgorithmResult(
            z=z,
            phi=phi_init,
            diagnostics=[],
            termination=RunTermination(reason="completed", iteration=0),
        )


register_algorithm(MyAlgorithm())
```

For a built-in, import `MyAlgorithm` and call `register_algorithm(MyAlgorithm())`
in `bcd_lab.algorithms.__init__`. External applications can make the same call
during startup. Then use `algorithm: {name: my_algo, params: {...}}` in a config. Each algorithm owns
construction and validation of its private run configuration; it should not
parse YAML or write result files itself.

Algorithms that support scheduled checkpoints also implement
`CheckpointableAlgorithm`. Their capture method returns a versioned
`AlgorithmCheckpoint` with an opaque payload, and their restore method rebuilds
opaque algorithm-owned state. The experiment runner never inspects
fields inside that payload. `run_bcd_iterations` is a BCD
lifecycle using BCD state and configuration; a different solver should own its
loop unless and until two implementations justify a smaller shared lifecycle.

An algorithm that wants its retained state exposed to generic result consumers
may additionally implement `SelectedStateExporter`. The exporter receives the
opaque selected checkpoint and returns strict JSON data. BCD currently exports
the selected link payload and the min/max of its reconstructed inner-product
matrix. The runner never inspects checkpoint internals. Plot renderers should
consume these exports rather than importing algorithm state types.

Successful BCD runs contain only finite observations, factors, centers,
predictions, gradients, and objectives. Numerical divergence raises
`BcdNumericalError` with the failed quantity and iteration; adjust the model,
initialization, or step sizes rather than treating a partial run as a result.

### Add an optimizer

Fixed-coordinate NumPy optimizers live under `bcd_lab.optimizers`. Register a
builder that accepts the configured parameter mapping and target array and
returns a fresh `ArrayOptimizer`. The instance exposes its learning rate, turns
gradients into updates, and owns a versioned checkpoint payload. Duplicate
names are rejected.

BCD selects its factor optimizer under `algorithm.params.z_optimizer`.
Built-in choices are `gradient_descent` and `adam`:

```yaml
algorithm:
  name: bcd
  params:
    z_optimizer:
      name: adam
      params:
        learning_rate: 1.0e-3
        beta1: 0.9
        beta2: 0.999
        epsilon: 1.0e-8
    phi_update:
      learning_rate: 1.0e-6
    regularization:
      alpha: 0.0
      lambda_: 0.0
      mu: 0.6
```

Adam preconditions only the Z data-gradient update. BCD then applies its
factor-balancing regularizer with the configured optimizer learning rate and
performs row projection. Phi does not use this optimizer boundary: dynamic
grids and Gaussian RKHS expansions change parameter coordinates, so adaptive
moments require an explicit model-owned transport policy.

### Add an objective

Objectives are differentiable data-fit terms. They receive observed values and
predictions and return both a scalar value and `d(value) / d(predictions)`.
They do not parse YAML, write files, or own algorithm-specific regularization.

```python
import numpy as np

from bcd_lab.losses import register_objective


def value(observed, predictions, params):
    weight = float(params.get("weight", 1.0))
    return weight * float(np.sum((predictions - observed) ** 2))


def prediction_gradient(observed, predictions, params):
    weight = float(params.get("weight", 1.0))
    return 2.0 * weight * (predictions - observed)


register_objective(
    "weighted_squared",
    value=value,
    prediction_gradient=prediction_gradient,
    squared_error_scale=None,
)
```

After importing the registration module, select it with
`objective: {name: weighted_squared, params: {weight: 2.0}}`. BCD owns and
adds its factor/link regularizers separately from this data-fit value.
The optional `squared_error_scale` capability should only be declared when the
objective is exactly `scale * sum((predictions - observed) ** 2)`. It enables
the closed-form `phi_sharp` diagnostic; objectives without that capability
return unavailable `phi_sharp` metrics rather than a misleading result.

### Add an initializer

Implement one of the `ZInitializer`, `PhiInitializer`, or `JointInitializer`
protocols in `src/bcd_lab/initializers/`. Import and register the implementation
once in `initializers/__init__.py`, then reference its registered name under
`initialization:`.

Separate initialization requires both `z` and `phi` entries. The built-in phi
initializer `none` deliberately constructs no function object and is valid for
BCD only with `link.kind: identity`; it never triggers an implicit zero-model
fallback.

### Add a function model

Function models implement only the capabilities they genuinely provide:

- `ScalarFunction` evaluates arrays.
- `CopyableFunction` additionally produces a detached function instance for
  initialization and diagnostic snapshots.
- `DifferentiableFunction` additionally evaluates the input derivative.
- `TrainableFunction` fits parameters, computes a residual-weighted update
  direction, applies it additively, and scales parameters for regularization.
- `ProjectableFunction` applies configured constraints in place.
- `BallRadiusResolver` optionally resolves a projection-ball radius from typed
  dataset truth without exposing model configuration or parameter storage to
  the algorithm.
- `SerializableFunctionModel` publishes an artifact payload.
- `CheckpointableFunction` publishes and restores its versioned mutable state.

A fixed link normally implements scalar evaluation, differentiation, copying,
and serialization only. It must not add no-op training or projection methods.
BCD accepts fixed links as-is; a trainable BCD link must implement both
`TrainableFunction` and `ProjectableFunction`. Register its builder with
`register_function_model(name)`. Builders return `ScalarFunction`, while each
algorithm checks any stronger capabilities it requires.
Checkpoint state is separate from presentation-oriented `to_payload()` output.
A checkpointable model implements `checkpoint_state()` and
`restore_checkpoint_state()`; the registered builder creates the correctly
configured empty instance before restoration. This lets new representations
round-trip without adding model-specific fields to the runner.

### Add a dataset loader

1. Implement `DatasetLoader` protocol in `src/bcd_lab/data/`
2. Import the loader in `data/__init__.py`
3. Call `register_data_loader(MyLoader())` there with the other built-ins
4. Use in config: `data.source: my_loader`

The returned `DatasetBundle` owns aligned train/validation/test sample sets and
the resolved `n_rows`, `n_cols`, and `rank`. Put reusable optional information
in typed `DatasetCapabilities`: function references, a function-evaluation
domain, linked truth, split statistics, and ID mappings. Keep loader-specific source provenance in
`metadata`; core algorithms and diagnostics do not interpret it.

The registry calls `DatasetBundle.validate()` before returning any built-in or
external dataset. It rejects invalid dimensions, split alignment, index bounds,
non-finite values, truth shapes, split counts, and ID mappings with loader
context. Loader code owns only source-specific policies.

### Add a synthetic generating link

Implement `GeneratingLinkBuilder` from `bcd_lab.data`. Its `validate_params`
method validates component-owned configuration during experiment preflight;
its `build` method receives a `GeneratingLinkContext` containing clean sampled
latent values, the dataset dtype, and an explicit NumPy generator. Return a
`ScalarFunction`, then register the builder once:

```python
from bcd_lab.data import register_generating_link

register_generating_link("my_truth", MyTruthBuilder())
```

Select it with
`data.generating_link: {name: my_truth, params: {...}}`. The synthetic loader,
not the experiment runner or BCD, applies it before noise and publishes truth
capabilities. Builders must reject unknown parameters and return finite values
over the generated latent matrix.

### Add a diagnostic metric

Diagnostic families live in `src/bcd_lab/diagnostics/`. Implement the
`DiagnosticFamily` protocol, import the class into `diagnostics/__init__.py`,
and register one instance there. Return numeric or textual metric values;
return `None` only when a requested metric is genuinely unavailable for the
published snapshot. Let computation defects raise so the configured
`on_error` policy can either stop the run or record them with context. New
families reach raw diagnostic records without runner changes. Every finite
numeric series also appears generically under `summary.metrics`; plot labels do
not decide scientific artifact contents.

### Add a plot renderer

Implement a renderer callable accepting one Matplotlib axis, one normalized
plot instance, and the shared rendering context. Register a
`PlotRendererDefinition` with its name, callable, and `normalize_options`
function. The normalizer receives the raw YAML mapping and plot label and
returns validated renderer-specific fields. When needed, declare
`supports_overlay`, `legend_policy`, or `raster_dpi`. Built-ins live in focused
modules under `bcd_lab.plotting.renderers` and register once in
`bcd_lab.plotting.__init__`; external code registers before plotting.

When a line renderer uses `trial_pick: random`, configure an integer
`trial_seed` on that plot or at the top of the `plot` section. Random trial
selection never falls back to process entropy.

### Loading external registrations

Built-ins are imported automatically in deterministic package order. External
components are not discovered automatically. With the installed command, pass
each registration module before the subcommand:

```bash
bcd-lab --plugin my_package.registrations run path/to/experiment.yaml
```

The module must call `register_algorithm`, `register_data_loader`, or another
registration function at import time. Repeat `--plugin` for multiple modules.
Do not assume a YAML component name imports third-party code by itself.
