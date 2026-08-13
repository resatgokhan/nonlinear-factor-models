# Contributing

`bcd-lab` is intended to be a readable research codebase for experimenting
with BCD-style matrix factorization algorithms, datasets, objectives, and
initialization strategies.

## Values

- Prefer readable modular code over clever abstractions.
- Keep additions narrow and remove dead code quickly.
- Respect existing extension patterns instead of adding special cases.
- Keep public APIs stable once they are introduced.
- Make experiments reproducible through explicit configs and seeds.

## Contribution Rules

- Add new algorithms through the algorithm registry.
- Add new initialization strategies through the initializer registry.
- Add new datasets through the data loader registry.
- Add synthetic truth functions through the generating-link registry.
- Add new losses/objectives through the objective registry.
- Add diagnostics, function models, plot renderers, and preprocessors through
  their corresponding registries.
- Do not hard-code new variants inside the experiment runner.
- Do not commit generated results, downloaded datasets, plots, profiles, or
  benchmark output folders.
- Keep domain-specific workflows separate from generic core
  code unless the abstraction is clearly reusable.


## Shared Runtime Objects

The algorithm-neutral runtime boundary consists of:

- `DatasetBundle`
- `AlgorithmResult`
- `AlgorithmCheckpoint`
- `RunSnapshot` and `DiagnosticRequest`
- the composable function capabilities (`ScalarFunction`,
  `DifferentiableFunction`, `TrainableFunction`, `ProjectableFunction`, and the
  optional `BallRadiusResolver`)
- the `Algorithm`, `CheckpointableAlgorithm`, dataset-loader, and initializer
  protocols
- `ArrayOptimizer` and its component-owned checkpoint state

These interfaces are used across packages, so changes should be explicit and
covered by focused tests. They are not yet promised as a separately versioned
stable public API. If an algorithm needs extra information, use a clearly
named field or documented metadata key; do not silently change a shared
object's semantics for one implementation.

`ModelConfig`, `AlgorithmContext`, `SolverState`, and
`run_bcd_iterations` belong to BCD. A new algorithm owns the
type returned by its `build_config` method and may own a different state and
lifecycle. Do not force unrelated solvers through the BCD loop merely to reuse
control flow. If a genuinely shared lifecycle is introduced later, extract only
the minimal scheduling and result contract required by two implementations.

Algorithm-private state remains private: publish a detached `RunSnapshot`, and
organize algorithm-specific diagnostic payloads as small family-owned
publishers so unrequested computations remain lazy.
Scheduled checkpoints require an algorithm-owned capture callback. Stateful
subcomponents such as function models and optimizers must publish and restore their own
versioned payload rather than exposing representation-specific attributes to
the loop or experiment runner.

Model selection is evaluated only on completed diagnostic records. Algorithms
supporting `retain_checkpoint: true` capture the exact observed state; strict
improvement replaces it and ties keep the earlier state. To expose selected
state to plots or tables, implement `SelectedStateExporter` and return a small
strict-JSON mapping. Keep the checkpoint opaque to the runner and keep
representation-specific conversion in the algorithm or owning component.

## Adding An Algorithm

An algorithm should:

- implement the `Algorithm` protocol in `bcd_lab.algorithms.registry`
- expose a non-empty `name`
- implement `build_config` and return its own validated configuration type
- implement the keyword-only `run` signature for that private configuration,
  `DatasetBundle`, `Objective`, optional `z_init` and `phi_init`, optional
  `diagnostics_request`, optional `selection_policy`, and an explicit NumPy
  generator
- return an `AlgorithmResult`
- avoid reading YAML directly
- avoid writing files directly

The experiment runner owns orchestration and payload writing. Algorithms own
optimization and solver-specific validation only. Built-ins are imported and
registered explicitly in `bcd_lab.algorithms.__init__`.

## Adding An Optimizer

An optimizer under `bcd_lab.optimizers` applies updates to one fixed-shape
floating NumPy array. Implement `ArrayOptimizer`, provide a builder accepting
`(params, parameter)`, and register that builder by name. Each build must return
a fresh instance; evolving moments and counters must never be shared between
runs. Validate parameters in the builder and shape, dtype, and finite values at
the update and checkpoint boundaries.

The optimizer returns the update that its algorithm subtracts. The consuming
algorithm owns gradient calculation and ordering relative to regularization and
projection. BCD currently uses registered optimizers only for Z. Phi remains
model-owned because dynamic grids and changing RKHS bases do not provide stable
coordinates for adaptive moments.

## Adding A Dataset

A dataset loader should:

- return a `DatasetBundle`
- define train/validation/test semantics clearly
- populate reusable `DatasetCapabilities` fields for function references,
  function-evaluation domains, linked truth, split statistics, ID mappings, or
  mutable-source fingerprints
- keep source-specific provenance in `metadata`
- avoid global state and hidden downloads unless explicitly configured

Large raw datasets should not be committed. `load_dataset` applies the canonical
`DatasetBundle.validate()` boundary to built-in and external loaders. Loaders
remain responsible for source-specific policy and contextual errors.

## Adding A Synthetic Generating Link

A generating link belongs to data generation, not to BCD and not to the fitted
function-model selection. Implement `GeneratingLinkBuilder` under
`bcd_lab.data.generating_links`, validate all builder-owned parameters in
`validate_params`, and construct one `ScalarFunction` from the provided
`GeneratingLinkContext`. Use only its explicit generator; do not mutate global
random state.

Register the builder once with `register_generating_link`. The synthetic loader
applies the function before adding noise and publishes it through
`DatasetCapabilities.functions.truth` together with `linked_truth`. New
generators therefore require no runner, algorithm, or diagnostic branches.

## Adding An Objective Or Loss

Objectives should be named, registered, and documented. Algorithms receive an
`Objective` exposing value and `d(value) / d(predictions)` rather than importing
one global loss function. Algorithm-specific regularization does not belong in
the objective implementation.

Objective implementations should state:

- what prediction form they expect
- whether they use observed entries only or dense matrices
- whether lower values are always better
- whether they expose the exact squared-error scale used by `phi_sharp`

## Adding An Initializer Or Function Model

Implement the narrow protocol the component needs and register it explicitly in
the corresponding package `__init__.py`. Separate initialization requires both
Z and phi selections. `phi.name: none` is a deliberate no-function result and
is valid for BCD only with the identity link.

Function models should expose only genuine capabilities. BCD requires scalar,
copyable, and differentiable behavior for fixed links; trainable links must also
be projectable. Models that infer a projection-ball radius from dataset truth
implement `BallRadiusResolver`. Mutable models own their checkpoint payload and
restoration version.

Representation policies stay with the model. For example, the piecewise-linear
model separates its grid-bound policy (`fixed` or `input_quantiles`) from its
rebuild policy (`never`, `adaptive`, or `every_iteration`); BCD only supplies
the centers to the model update. If a model accepts structured parameters,
register its configuration validator with the builder so malformed YAML fails
during experiment preflight, before data loading or output creation.

## Adding A Diagnostic Family

Implement `DiagnosticFamily`, consume only `DiagnosticContext.snapshot` and
typed dataset capabilities, and register the family in
`bcd_lab.diagnostics.__init__`. Return `None` only for genuinely unavailable
metrics; defects should raise and flow through the configured `raise` or
`record` policy.

Function-comparison diagnostics must use the dataset's typed
`function_domain` when one is present. The loader owns construction of that
domain; diagnostics must not infer a competing range from a function model's
private grid.

New finite numeric metrics reach both raw diagnostic records and generic
`summary.metrics` without runner or plotting changes.

## Adding A Plot Type

Put each built-in plot family in one focused module under
`bcd_lab.plotting.renderers`. Its renderer receives one Matplotlib axis, the
normalized plot instance, and the shared rendering context. Register it once,
in deterministic order, in `bcd_lab.plotting.__init__`.

External applications register a `PlotRendererDefinition` during startup. Its
`normalize_options` callback validates renderer-specific YAML and returns the
normalized fields added to the plot entry. Declare `supports_overlay`,
`legend_policy`, and `raster_dpi` only when needed. Renderers should draw only
their panel:
record loading and metric transformation belong in `plotting.data`, grouping
and sizing in `plotting.layout`, legends in `plotting.legends`, and artifact
writing in `plotting.output`. Do not introduce a second plotting entrypoint or
write files from a renderer.

Every public plot type should have an end-to-end YAML-to-output test for at least
one meaningful renderer-owned option. Registration is duplicate-safe and the
service contains no built-in-name checks for these capabilities.

## Built-ins And External Registration

Built-ins are imported and registered once, in deterministic order, by their
package `__init__.py`. Duplicate names raise instead of replacing an existing
component.

External code may call the same public registration functions, but its module
must be imported before configuration preflight. The installed command provides
that boundary: `bcd-lab --plugin my_package.registrations run config`. Repeat
`--plugin` for independent modules. Import failures are explicit and duplicate
registrations still fail.

## Tests

Prefer small targeted tests:

- schema parsing tests for config changes
- smoke tests for new algorithms
- deterministic tests for initializers
- metric tests for evaluators
- one end-to-end sweep when adding or changing an extension boundary

Do not rely on long sweeps as the only validation.

Run the repository gates before submitting a change:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy
python3 -m pytest
python3 -m build --no-isolation
```
