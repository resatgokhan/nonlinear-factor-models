# Architecture

This document describes the current `bcd-lab` implementation. It is the place
to start when deciding where a change belongs; the implementation remains the
source of truth.

## Runtime flow

One experiment moves through a single composition path:

```text
YAML configuration
  -> experiments.loader parses and validates ExperimentSpec
  -> experiments.sweep expands complete configurations
  -> experiments.composition resolves registered components and preflights cases
  -> experiments.runner derives random streams and coordinates each trial
  -> data loader returns a validated DatasetBundle
  -> initializer builds algorithm inputs
  -> algorithm owns optimization, private state, and checkpoints
  -> algorithm publishes a detached RunSnapshot for diagnostics
  -> experiments.artifacts writes versioned result and checkpoint files
  -> plotting loads saved artifacts and dispatches registered renderers
```

`bcd_lab.experiments` is the composition root. It may depend on component
packages to assemble a run. Component packages must not depend on the runner,
parse YAML, or write experiment artifacts themselves.

## Package ownership

| Package or module | Owns | Does not own |
|---|---|---|
| `experiments` | schema, validation, sweeps, orchestration, artifact creation | optimization or component-private parameters |
| `algorithms` | solver registration, BCD configuration/state/lifecycle, snapshots | YAML, dataset loading, output paths |
| `initializers` | separate and joint construction of starting values | implicit algorithm fallbacks |
| `data` | loaders, synthetic generation, source provenance, preprocessing | solver policy |
| `function_models` | link capabilities, implementations, projection, model state | experiment orchestration |
| `optimizers` | fixed-coordinate array updates and versioned optimizer state | BCD policy or function-model coordinates |
| `diagnostics` | snapshot-based metric families and failure policy | live solver state |
| `plotting` | artifact loading, plot normalization/layout, renderer dispatch, PDF output | scientific summary selection |
| `losses.py` | registered prediction objectives | algorithm regularization |
| `types.py` | shared datasets, results, snapshots, and capability values | component-private state |
| `randomness.py` | deterministic named seed derivation | global RNG mutation |
| `checkpoints.py` | representation-neutral archive encoding | interpreting algorithm payloads |

## Dependency rules

- Registries are the public selection boundary. Adding a component must not add
  a name-specific branch to `experiments.runner`.
- Algorithms own the type produced by `build_config`. `ModelConfig`,
  `SolverState`, and `run_bcd_iterations` are BCD-specific, not universal solver
  abstractions.
- Dataset loaders return `DatasetBundle`; the registry validates its stable
  structural invariants immediately. Reusable optional information belongs in
  typed `DatasetCapabilities`, while loader-only provenance stays in metadata.
- Function models expose small capabilities. BCD checks only the capabilities a
  selected mode needs and does not inspect representation-specific fields.
- Optimizers transform fixed-shape array gradients into updates. Algorithms own
  their instances and ordering relative to regularization and projection.
- Diagnostics receive immutable, detached `RunSnapshot` values. They never read
  or mutate an algorithm's live state.
- Plotting consumes result artifacts. Renderer definitions own their YAML option
  normalization and rendering capabilities, so the service does not branch on
  built-in renderer names.
- Algorithms and components do not choose paths. Configuration-relative input,
  result, plot, and checkpoint paths are resolved at orchestration boundaries.
- Random work consumes an explicit seed or generator derived from the experiment
  seed. No component reseeds Python or NumPy global state.

## Public extension map

| To add | Implement and register | Registration package |
|---|---|---|
| Algorithm | `Algorithm` | `bcd_lab.algorithms` |
| Objective | `ObjectiveDefinition` callbacks | `bcd_lab.losses` |
| Z, phi, or joint initializer | matching initializer protocol | `bcd_lab.initializers` |
| Dataset source | `DatasetLoader` | `bcd_lab.data` |
| Function model | builder plus genuine model capabilities | `bcd_lab.function_models` |
| Fixed-coordinate optimizer | `ArrayOptimizer` builder | `bcd_lab.optimizers` |
| Diagnostic family | `DiagnosticFamily` | `bcd_lab.diagnostics` |
| Plot type | `PlotRendererDefinition` | `bcd_lab.plotting` |
| Raw-data preprocessor | `DataPreprocessor` | `bcd_lab.data.preprocessors` |

Built-ins are wired explicitly and deterministically in package `__init__.py`
files. External registration modules are imported before command dispatch with
repeatable `bcd-lab --plugin MODULE` options. Duplicate names fail and never
replace an existing definition.

Detailed implementation checklists and examples live in
[CONTRIBUTING.md](CONTRIBUTING.md). User-facing configuration and command
examples live in [USAGE.md](USAGE.md).

## Persisted boundaries

Completed trials write one versioned JSON artifact per expanded case and an
optional sibling checkpoint archive. Filenames include canonical configuration
identity, writes are atomic, and existing targets require explicit overwrite.
CSV inputs carry a content fingerprint so replay detects changed source data.
See [RESULT_ARTIFACTS.md](RESULT_ARTIFACTS.md) for the exact contract.

## Deliberately specific code

Not every numerical implementation should become a framework abstraction.
The BCD loop, BCD state, phi update, RKHS projections, dynamic-grid rules, and individual
plot renderers remain cohesive implementation modules. Extract a shared
abstraction only after another real component needs the same contract.
