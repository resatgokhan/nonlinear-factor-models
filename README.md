# bcd-lab

**bcd-lab** is a nonlinear matrix-factorization research framework. YAML configs
select registered algorithms, data loaders, initializers, and evaluation components.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]"
bcd-lab --help
bcd-lab list                                    # Bundled configs
bcd-lab run examples/quickstart                 # Run only
bcd-lab run examples/quickstart --plot           # Run and plot
bcd-lab plot examples/quickstart                # Plot saved results
python3 -m pytest tests/
```

See [USAGE.md](USAGE.md) for configs, data, sweeps, and plots, and
[RESULT_ARTIFACTS.md](RESULT_ARTIFACTS.md) for the versioned result/checkpoint format.

## Repository layout

```text
configs/                     # Self-contained experiment catalog (schema v2)
  examples/                  # Small learning-oriented runs
  figures/                   # Figure 1–4 studies
  real_data/                 # MovieLens and Jester runs
  studies/                   # Optimizer and tuning studies
finance/                     # Stock-return covariance experiment and configs
src/bcd_lab/
  algorithms/                # Solver configuration, state, lifecycle, registry
  data/                      # Loaders, generating_links/, and preprocessors/
  diagnostics/               # Metric families consuming detached snapshots
  experiments/               # YAML, validation, sweeps, composition, artifacts
  function_models/           # Link capabilities, models, checkpoints
  initializers/              # Factor and link initialization
  optimizers/                # Fixed-coordinate array optimizers
  plotting/                  # Plot services and registered renderers/
  checkpoints.py             # Representation-neutral checkpoint archive
  randomness.py              # Named deterministic random streams
  types.py                   # Shared datasets, results, snapshots
tests/                       # Unit, boundary, and smoke tests
data-raw/                    # Raw downloads (Git-ignored)
data/                        # Preprocessed data (Git-ignored)
```

## Architecture and extension rules

The matrix-factorization runner follows:

```text
YAML → validation and sweep preflight → registry resolution and named RNG streams
     → DatasetBundle and private algorithm config → initialization and optimization
     → detached diagnostic snapshots and optional validation-selected model
     → versioned results/checkpoints → plots from saved artifacts
```

`experiments` composes components; algorithms own optimization and private
parameters. YAML independently controls data, algorithm, objective,
initialization, link, regularization, diagnostics, and sweep axes. Sweep paths
can vary both component names and parameters across the full specification.

Add variants through registries rather than branches in the runner. Algorithms,
objectives, initializers, optimizers, loaders, generating links, diagnostics,
function models, renderers, and preprocessors are registered by name. Load
external registrations with repeatable `bcd-lab --plugin MODULE` options.
Algorithm, objective, initializer, optimizer, loader, generating-link, diagnostic,
and function-model registrations have tested execution paths.

Function models expose explicit evaluation, differentiation, training,
projection, distance/norm, and serialization capabilities; fixed links need no
training methods. Algorithms validate only the capabilities they use. Renderers
own YAML option normalization and overlay, legend, and raster capabilities.
Validation-based model selection retains one iteration's complete diagnostics,
checkpoint, and plotting exports without coupling the runner to BCD state.

## Example experiments

Tracked configs are runnable examples, not test fixtures:

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
| [Stock-return covariance](finance/README.md) | Dated-return panels, covariance estimation, and rolling portfolios | Smoke run or full experiment |

See [configs/README.md](configs/README.md) for the full matrix-factorization
catalog, workload, and publication/archive provenance.

Place official downloads in `data-raw/`, then run
`bcd-lab preprocess movielens-latest-small` or `bcd-lab preprocess jester-dataset-1`.
Preprocessors validate release dimensions and rating ranges and write clean
triple CSVs with fingerprinted provenance to `data/`, which the runner reads.
See [USAGE.md](USAGE.md#real-data-pipeline) for downloads and directory layout.

## Dataset extension contract

Every loader returns a `DatasetBundle` with required observation/shape fields
and typed optional `dataset.capabilities`:

| Capability | Contents |
|---|---|
| `functions` | Truth/initial links and diagnostic sample count |
| `function_domain` | Read-only evaluation grid and construction policy |
| `linked_truth` | Dense matrix after applying the truth link |
| `splits` | Train/validation/test counts and statistics |
| `id_mappings` | Source identifiers mapped to dense indices |
| `source_fingerprint` | SHA-256 and size of external input |
| `preprocessing` | Converter policy, raw/clean fingerprints, identifiers, counts |

Read stable fields directly; add typed capabilities for concepts shared by
components. Reserve `metadata` for loader-specific provenance. Missing optional
capabilities make diagnostics unavailable; required capabilities (such as an
inferred link-ball radius) are checked before initialization. The registry
validates dimensions, split alignment/counts, indices, finite values, truth
shapes, and ID mappings before consumers see a bundle.

Synthetic truth uses `data.generating_link: {name, params}`; the fitted model
uses `link`. Generators include `identity`, `sigmoid`, `exponential`, `tanh`, and
`random_monotone_rkhs`. The last defaults to a strictly increasing degree-four
polynomial fitted in a Gaussian RKHS and exposes truth through these capabilities.

Optional `data.evaluation_domain: {central_mass: 0.95, grid_count: 1000}` derives
a shared diagnostic grid from the central 95% of the latent truth matrix.

## Documentation and contributing

- [USAGE.md](USAGE.md) — installation, YAML, commands, data, and plots
- [configs/README.md](configs/README.md) — experiment catalog and provenance
- [ARCHITECTURE.md](ARCHITECTURE.md) — package ownership and dependency boundaries
- [CONTRIBUTING.md](CONTRIBUTING.md) — registered component extension contracts
- [RESULT_ARTIFACTS.md](RESULT_ARTIFACTS.md) — result and checkpoint formats

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
