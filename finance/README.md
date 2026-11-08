# Stock-return covariance experiment

Estimate stock-by-stock covariance from a complete log-return matrix (dates ×
stocks), fit BCD on chronological training windows, and evaluate portfolios on
subsequent observations. Algorithm, link, initialization, estimator, and portfolio
rule are independent choices. Composition lives in `bcd_lab.finance`; loading in
`bcd_lab.data.finance`. See [Methodology](METHODOLOGY.md) for the model, covariance
formulas, window settings, evaluation metrics, and assumptions.

## Run

Install with `python3 -m pip install -e '.[dev]'`. Supply panels locally; data and
outputs are Git-ignored. Configs specify filenames, stock counts, and SHA-256
fingerprints; the import helper verifies digests before copying.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 finance/import_panels.py /path/to/return-panels
python3 -m bcd_lab.finance run finance/configs/smoke.yaml
python3 -m bcd_lab.finance run finance/configs/log_panel_N50.yaml
# Other full presets: log_panel_N100.yaml, log_panel_N200.yaml
python3 -m bcd_lab.finance report results/finance/log_panel_N50
```

Options: `--period-limit 2`, `--methods sample linear_shrinkage`, `--output-dir PATH`.
Nonempty destinations require `--overwrite`. Failures retain the manifest and
completed records; there is no implicit resume.

Artifact names must be unique, ignoring case. A fitted method `foo` conflicts
with `foo_checkpoints`: both write `foo_checkpoints.npz`. Collisions fail before
fitting or writing, even with `--overwrite`.

## Input format

CSV: `Date`, then unique stock identifiers. Dates are sorted and must be unique;
missing/nonfinite values fail validation. For other datasets, set the path
(relative to the config), stock count, fingerprint, and window schedule.
The full presets use 50/100/200 stocks, 250 training observations, and 107
nonoverlapping 21-observation evaluation windows.

## Configuration, artifacts, and tests

YAML uses `schema_version: 1`. `panel`, `window`, and `evaluation` define inputs
and evaluation; `model_defaults` supplies shared settings. Each `methods` entry
selects a model, estimator, and portfolio. Changing a link/algorithm selector
replaces that component's default block.

`run.json` records config, fingerprint, ticker order, calendar windows, dependency
provenance, and running/failed/complete status. Each window/method saves covariance,
weights, daily scores, and fit diagnostics as JSON and pickle-free NPZ. Fitted
models also save fitted/residual returns and versioned factors/link/optimizer
checkpoints. Writes are atomic; reports recompute summaries and figures from saved
scores without refitting.

`run` automatically generates the report; `report` regenerates it for a completed
run, replacing these files:

| Output | Contents |
|---|---|
| `summary.csv` | Per-method annualized return score (AV), volatility (SD), Sharpe (SR), observation/window counts, and timings |
| `metrics.png`, `metrics.pdf` | Charts of AV, SD, and SR |

Reports validate record completeness and date alignment and reject duplicate
evaluation dates. Scores pool daily observations across windows; definitions and
interpretation are in [Methodology](METHODOLOGY.md#portfolio-evaluation).

```sh
PYTHONPATH=src python3 -m pytest -q tests/test_finance.py tests/test_finance_covariance.py tests/test_finance_model.py
```
