# bcd-lab

**bcd-lab** is an experimental framework for BCD-style nonlinear matrix
factorization research. It separates algorithm implementation, data loading,
initialization, and experiment orchestration into registry-driven components
selected from YAML configs.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Show the command interface
bcd-lab --help

# Run a specific experiment
bcd-lab run examples/quickstart

# Run it and then generate its configured plots
bcd-lab run examples/quickstart --plot

# Plot results from an earlier run without rerunning the experiment
bcd-lab plot examples/quickstart

# See all bundled experiment configs
bcd-lab list

# Run tests
python3 -m pytest tests/
```

`bcd-lab --help` shows the direct command interface. See
**[USAGE.md](USAGE.md)** for the full workflow — configs, real data, sweeps,
and plots. Completed trials use the versioned, replay-oriented
**[result artifact contract](RESULT_ARTIFACTS.md)**.

---

## Repository Layout

```
bcd-lab/
│
├── configs/                    # Public experiment catalog (schema v2)
│   ├── examples/               # Small learning-oriented runs
│   ├── figures/                # Figure 1–4 publication-aligned studies
│   ├── real_data/              # Primary MovieLens and Jester runs
│   ├── studies/                # Optimizer and tuning investigations
│   └── README.md               # Workload, status, provenance, commands
│
├── src/bcd_lab/
│   ├── algorithms/             # BCD, lifecycle, state, snapshots, registry
│   ├── data/                   # Dataset loaders and generating links
│   │   ├── generating_links/   # Synthetic truth-function registry
│   │   └── preprocessors/      # Raw → clean CSV pipeline
│   ├── diagnostics/            # Modular metric families
│   ├── experiments/            # Schema, validation, composition, artifacts
│   ├── function_models/        # Link capabilities, models, checkpoints
│   ├── initializers/           # Z and phi initialization strategies
│   ├── optimizers/             # Fixed-coordinate array optimizers
│   ├── plotting/               # Plot services + registry-driven renderers
│   │   └── renderers/          # One focused module per plot family
│   ├── checkpoints.py          # Representation-neutral checkpoint archive
│   ├── randomness.py           # Named deterministic random streams
│   └── types.py                # Shared datasets, results, snapshots
│
├── tests/                      # Unit, boundary, and smoke tests
├── ARCHITECTURE.md             # Package ownership and dependency boundaries
├── CONTRIBUTING.md             # Extension contracts and contribution rules
├── RESULT_ARTIFACTS.md         # Versioned result/checkpoint contract
├── USAGE.md                    # Configuration and command reference
│
├── data/                       # Clean preprocessed CSV files  [gitignored]
└── data-raw/                   # Raw downloaded dataset files  [gitignored]
```

## Architecture at a glance

One experiment follows this path:

```text
YAML
  → parse and validate ExperimentSpec
  → expand sweep axes and preflight every case
  → resolve registered data, objective, initialization, and algorithm components
  → derive named data, initialization, and algorithm random streams
  → load DatasetBundle and build the algorithm's private configuration
  → initialize and run the algorithm
  → publish detached RunSnapshot values to diagnostics
  → retain an optional validation-selected model at an observed iteration
  → save versioned JSON results and component-owned checkpoints
  → load result artifacts into the plotting services
```

`experiments` is the composition root. It owns YAML, sweep orchestration,
preflight, and artifacts, but not optimization or component-private parameters.
Algorithms own configuration and state; data loaders own source-specific
loading; initializers construct starting values; diagnostics consume detached
snapshots; plotting reads saved artifacts rather than live solver state.

## Design Principles

- **Registry-driven** — algorithms, objectives, initializers, optimizers, data
  loaders, synthetic generating links, diagnostic families, function models,
  plot renderers, and preprocessors are registered by name. Experiment
  composition resolves components through those registries rather than
  solver- or dataset-specific branches.
- **Config-first** — experiments are fully described by YAML. Configs declare
  the data source, algorithm, objective, initialization, link function, regularization,
  diagnostics, and sweep axes.
- **Composable components** — algorithm, initialization, data source, and
  evaluation are separate concepts. Sweep paths are applied to the complete
  experiment specification, so component names and their parameters can vary
  independently.
- **Coherent model selection** — a configured diagnostic metric can retain one
  exact iteration's full diagnostic record, algorithm checkpoint, and generic
  plotting exports without coupling the runner to BCD state.
- **Two-tier data** — `data-raw/` holds raw downloads; `data/` holds clean
  triple CSVs written by preprocessors. The experiment runner reads only from
  `data/`.
- **Narrow changes** — new variants plug into registries, not into `if/elif`
  chains inside the runner.

Function models follow the same composition rule internally: scalar
evaluation, differentiation, training, projection, native distance/norm, and
serialization are explicit capabilities. Fixed links expose no meaningless
training operations, and algorithms validate only the stronger capabilities
they actually consume.

## Example experiments

The tracked configurations are examples rather than test fixtures:

| Config | Demonstrates | Expected workload |
|---|---|---|
| `examples/quickstart` | Deterministic matrix completion and PDF generation | Seconds |
| `examples/synthetic_link_recovery` | Learned piecewise-linear link with truth diagnostics | Under a minute |
| `figures/figure_01_sigmoid_noise` | Known-link noise study | Long research run |
| `figures/figure_02_joint_recovery_noise` | Joint-learning noise study | Long research run |
| `figures/figure_03_identity_sample_size` | Identity-link sample-size study | Long research run |
| `figures/figure_04_joint_regularization` | Joint-learning regularization study | Long research run |
| `real_data/movielens_learned_link` | Full latest-small learned-link run | Long after preprocessing |
| `real_data/jester_learned_link` | Full Jester learned-link run | Long after preprocessing |

The complete catalog, including exploratory optimizer studies and documented
publication/archive discrepancies, is in [configs/README.md](configs/README.md).

For the exact real-data releases, run
`bcd-lab preprocess movielens-latest-small` or
`bcd-lab preprocess jester-dataset-1` after placing the official downloads under
`data-raw/`. See [USAGE.md](USAGE.md#real-data-pipeline) for the precise layout
and download commands. These conversions validate authoritative dimensions and
rating ranges and emit a fingerprinted provenance sidecar next to each CSV.

## Dataset extension contract

Every loader returns a `DatasetBundle`. Required observations and shape fields
are direct attributes. Reusable optional features are typed under
`dataset.capabilities`:

- `functions`: reference truth/initial link functions and diagnostic sample count
- `function_domain`: a read-only grid and its documented construction policy
- `linked_truth`: the dense matrix after applying the reference link
- `splits`: train/validation/test counts and split statistics
- `id_mappings`: source identifiers mapped to dense row/column indices
- `source_fingerprint`: SHA-256 and size identity for mutable external input
- `preprocessing`: converter policy, raw-source fingerprints, identifier
  summaries, counts, and the verified clean-output fingerprint

Algorithms and diagnostics should read these stable fields directly. Add a new
typed capability when more than one component needs a concept; reserve
`metadata` for loader-specific provenance or details that core code never
interprets. Missing optional diagnostic capabilities produce unavailable
metrics rather than failing a run. Features that require a capability, such as
an inferred link-function ball radius, validate it before initialization.

The registry validates every returned bundle before algorithms or diagnostics
consume it: dimensions, aligned splits, index bounds, finite values, optional
truth shapes, split counts, and ID mappings share one canonical boundary.

Synthetic data distinguishes the function that generates observations from the
function model fitted by BCD. Configure the former under
`data.generating_link: {name, params}` and the latter under `link`. Built-in
generators are `identity`, `sigmoid`, `exponential`, `tanh`, and
`random_monotone_rkhs`. The random paper generator draws a strictly increasing
degree-four polynomial by default, fits it in a Gaussian RKHS, and publishes
the resulting truth through the same typed dataset capabilities.

Synthetic experiments may additionally configure
`data.evaluation_domain: {central_mass: 0.95, grid_count: 1000}`. The loader
derives the grid from the central 95% of the latent truth matrix and publishes
it with the dataset. Function and linked-truth diagnostics then share one
data-owned evaluation interval instead of choosing representation-specific
ranges.

## Extension status

Algorithms, objectives, initializers, fixed-coordinate optimizers, data loaders,
generating links, diagnostic families, and function models have tested
registry-to-execution paths. Plot renderer definitions additionally own YAML
option normalization and the small overlay, legend, and raster capabilities
required by the service. External registration modules can be loaded explicitly
with repeatable `bcd-lab --plugin MODULE` options before command dispatch.

## Contributing

Start with [ARCHITECTURE.md](ARCHITECTURE.md) for the current module boundaries,
then use [CONTRIBUTING.md](CONTRIBUTING.md) for the extension contracts.

## Documentation

- [USAGE.md](USAGE.md) — installation, YAML, commands, data, plots, and examples
- [ARCHITECTURE.md](ARCHITECTURE.md) — current package structure and ownership
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add each registered component
- [RESULT_ARTIFACTS.md](RESULT_ARTIFACTS.md) — saved result and checkpoint format

## License

This project is available under the [MIT License](LICENSE).

## Citation

This repository supports the methods described in:

> Yutong Chao, Resat Gökhan, Jalal Etesami, and Ali Habibnia. “Learning
> Nonlinear Factor Models with Unknown Monotone Links from Incomplete and
> Noisy Data.” arXiv:2605.26271, 2026.

- Paper: [arXiv:2605.26271](https://arxiv.org/abs/2605.26271)
- Machine-readable citation metadata: [CITATION.cff](CITATION.cff)

```bibtex
@article{chao2026learning,
  title   = {Learning Nonlinear Factor Models with Unknown Monotone Links from Incomplete and Noisy Data},
  author  = {Chao, Yutong and Gökhan, Resat and Etesami, Jalal and Habibnia, Ali},
  journal = {arXiv preprint arXiv:2605.26271},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.26271}
}
```
