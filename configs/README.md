# Experiment configuration catalog

Configurations are grouped by purpose. The canonical CLI key is the path below
`configs/` without `.yaml`; for example:

```bash
bcd-lab run examples/quickstart --plot
bcd-lab run figures/figure_02_joint_recovery_noise
bcd-lab run real_data/movielens_learned_link --plot
```

Every file is self-contained. Scientifically relevant dimensions, random seed,
initialization, optimization, regularization, diagnostics, and sweep values are
explicit rather than hidden behind cross-file inheritance.

## Catalog

| Config key | Purpose | Status | Expected workload |
|---|---|---|---|
| `examples/quickstart` | Small identity-link first run | Verified example | Seconds |
| `examples/synthetic_link_recovery` | Learned monotone-link recovery | Verified example | Under a minute |
| `figures/figure_01_sigmoid_noise` | Known sigmoid link across noise levels | Publication-aligned adaptation | Long |
| `figures/figure_02_joint_recovery_noise` | Joint factor/link recovery across noise levels | Publication-aligned adaptation | Long |
| `figures/figure_03_identity_sample_size` | Identity-link recovery across sample sizes | Publication-aligned adaptation | Long |
| `figures/figure_04_joint_regularization` | Minimum factor error across alpha | Publication-aligned adaptation | Long |
| `real_data/movielens_learned_link` | Full MovieLens latest-small monotone learned link | Archived-parameter adaptation | Long |
| `real_data/jester_learned_link` | Full Jester Dataset 1 monotone learned link | Publication-parameter adaptation | Long |
| `studies/z_learning_rate` | Repeated factor learning-rate comparison | Exploratory | Short |
| `studies/movielens_adam_learning_rate` | Adam learning-rate sweep on MovieLens 100K | Exploratory | Short to medium |
| `studies/movielens_optimizer_comparison` | Gradient descent versus Adam | Exploratory | Short to medium |

The four `figures/` files correspond to Figures 1–4 in
[Learning Nonlinear Factor Models with Unknown Monotone Links from Incomplete and Noisy Data](https://arxiv.org/abs/2605.26271).
They use the canonical deterministic RNG streams in this repository, so they
are described as publication-aligned adaptations until numerical parity against
frozen archived outputs is documented.

## Real-data inputs

Prepare the exact inputs before running the primary real-data configs:

```bash
bcd-lab preprocess movielens-latest-small
bcd-lab preprocess jester-dataset-1

bcd-lab run real_data/movielens_learned_link --plot
bcd-lab run real_data/jester_learned_link --plot
```

MovieLens uses the archived rank, gradient-descent rates, 1,600-iteration
horizon, and configured projection radius `0.5`; the publication reports the
resulting beta as `0.496`. Jester's publication table reports beta `3.537`, which
is explicit in its config. The archived Jester YAML instead inferred its radius
and the saved run recorded approximately `10.266`. This discrepancy remains
visible here and must be resolved before claiming exact reproduction.

The figure configs follow the publication's joint-learning lambda `0.5`. The
archived joint-noise YAML records `0.0`; this is likewise an open provenance
discrepancy, not a hidden compatibility rule.

## Archived source mapping

| Public config | Archived configuration evidence | Archived result evidence |
|---|---|---|
| `figures/figure_01_sigmoid_noise` | `sigmoid_noisy_scale.yml` | `results/sigmoid_noisy_scale/` |
| `figures/figure_02_joint_recovery_noise` | `big_speed_noise_std.yml` | `results/big_speed_noise_std/` |
| `figures/figure_03_identity_sample_size` | `identity_m.yml` and its recorded full sweep | `results/identity_m/` |
| `figures/figure_04_joint_regularization` | `sweep_alpha.yml` for the axis; `sweep_alpha_speed.yml` for dimensions and horizon | `results/sweep_alpha*/` |
| `real_data/movielens_learned_link` | `movie_lens_speed_monotone_increasing_clamp.yml` | `results/movie_lens_speed_monotone_increasing_clamp/` |
| `real_data/jester_learned_link` | `jester_speed_monotone_increasing_clamp_pm10.yml` | `results/jester_speed_monotone_increasing_clamp_pm10/` |

The archived repository is supporting evidence maintained separately; none of
its generated artifacts are copied into this public repository.

Generated data, results, checkpoints, and plots remain outside Git.
