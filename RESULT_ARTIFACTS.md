# Result artifact contract

Each completed trial writes one JSON result artifact. `artifact_version` is the
version of this file contract; version 2 has these canonical sections:

- `experiment_spec`: an exact one-run `ExperimentSpec`, including registry
  names, component parameters, defaults, effective seeds, and the dataset shape
  resolved after loading. It has `trials: 1` and no sweep axes;
- `run`: source-trial context, effective named component seeds, UTC start time,
  wall-clock duration, and the algorithm termination reason/iteration;
- `provenance`: the `bcd-lab`, Python, and key dependency versions together
  with the Git revision and dirty state when the package is run from a checkout;
- `diagnostics`, `metadata`, and `summary`: the scientific output consumed by
  the plotting pipeline.
- `sweep_parameters`: the exact canonical sweep-path assignments for this run,
  used directly by plotting axes and legends.
- optional `selection`: the selected metric, direction, iteration, value, the
  complete diagnostic record from that same iteration, the selected checkpoint
  filename, and small algorithm-owned JSON exports for tables and plots.

Configuration is stored only in `experiment_spec`. Its scalar `seed` is the
effective root seed for this trial, so replaying its one trial derives the same
streams. `run.seeds` records that root plus the effective named data,
initialization, and algorithm seeds for inspection; `run.trial` identifies
which trial in the source experiment produced the artifact but does not affect
replay.
The `plot` section is retained as run-neutral presentation configuration so the
complete YAML input is not silently truncated.

`bcd_lab.experiments.artifacts.deserialize_experiment_spec` reconstructs and
validates the saved specification. Expanding the reconstructed specification
produces exactly one case, which can be preflighted and run normally.

CSV runs record a SHA-256 digest and byte size in both the resolved data
configuration and artifact provenance. Replaying that resolved configuration
verifies the file before parsing and fails if content at the path changed.
Other external loader types should expose an equally stable typed source
identity when their inputs are mutable.

Artifact serialization is strict JSON. Finite numerical arrays used by compact
scientific exports are converted to nested JSON arrays; non-finite floats and
unsupported runtime objects are rejected instead of being converted to lossy
strings.
Each filename ends in a canonical content-hash prefix, so typed sweep values do
not collide through display slugs. The runner plans every name before creating
output, rejects existing JSON or checkpoint targets by default, and replaces
them only when `overwrite=True` is explicit. JSON and NPZ writes use sibling
temporary files plus atomic replacement; a failed write preserves the previous
artifact.

## Checkpoint archives

When checkpoint saving is enabled, the JSON result has a sibling
`<run_name>_checkpoints.npz` archive. Its manifest records the archive format,
iteration, algorithm name, algorithm-state version, component-state versions,
and elapsed time. Numerical arrays remain typed NPZ members; nested payload
structure is encoded in the manifest without pickle.

The archive service treats algorithm payloads as opaque mappings. Restore by
loading the archive with `bcd_lab.checkpoints.load_checkpoint_archive` and
passing the selected `AlgorithmCheckpoint` to the registered algorithm's
`restore_checkpoint` method together with the resolved run configuration.
Unsupported versions fail explicitly rather than being guessed or partially
loaded.

When a selection policy retains a model, the runner also writes
`<run_name>_selected_checkpoint.npz`. It contains exactly one checkpoint and is
written even when scheduled checkpoint saving is disabled. For BCD this opaque
state contains the matching Z, phi, and factor-optimizer state. The sibling
JSON's `selection.diagnostics` and `selection.exports` are derived from that
same observed iteration; `selected_link` plots read the exported function
payload rather than decoding the checkpoint.
