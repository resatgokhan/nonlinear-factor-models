# Stock-return covariance methodology

This experiment estimates covariance **between stocks**, using dates as repeated
observations. It produces an N × N covariance matrix, not a time-by-time matrix.
See [README](README.md) for commands and outputs. Formulas below describe the
implementation in [covariance.py](../src/bcd_lab/finance/covariance.py) and
[evaluation.py](../src/bcd_lab/finance/evaluation.py); code blocks remain readable
in Markdown viewers without equation rendering.

## Data and windows

Let `R` be a T × N matrix of log returns: T dates, N stocks. Inputs must be complete
and finite, with unique dates and stock identifiers. Dates are sorted; no
imputation, scaling, or precentering is applied before fitting. Covariance
calculations center each stock's observations.

| Full-preset setting | Value |
|---|---|
| Panels | 50 / 100 / 200 stocks; 5,032 rows, 2005-01-04–2024-12-31 |
| Training / evaluation / rebalance | 250 / 21 / 21 observations |
| Schedule | Starts on/after 2015-01-01; 107 evaluation windows before 2025-01-01 |
| First training window | 2015-01-02–2015-12-29 |
| First evaluation window | 2015-12-30–2016-01-29 |

Each window fits models and estimates covariance using training observations
only. Weights are then evaluated on the following observations. Training windows
can overlap; evaluation windows cannot. Insufficient history is rejected.

## BCD fit

With rank k, `U` is T × k and `V` is N × k. BCD uses a squared-error objective to
fit `F = phi(U @ V.T)`, where the scalar link acts elementwise. Residuals are
`E = R - F`. The full nonlinear presets use:

- `piecewise_linear_rkhs`: 40 interpolation points, Gaussian width 1. Grid bounds
  use input quantiles 1%–99%, padded by `max(0.25 * span, 0.5)`, and are rebuilt
  before each link update and its residual evaluation.
- SVD factors, zero initial link, seed 0 for each window, float64, and a fixed
  500-iteration budget (`tol: 0`). Factor projection uses a training-window
  empirical incoherence bound; factor balancing regularization `lambda_` is 0.
- Monotonicity, output clamping, and link-ball projection disabled. A fixed
  identity link and the trainable-link initializer `identity_phi` are available.

| Stocks | Rank | Link learning rate | Factor learning rate | Link regularization |
|---|---:|---:|---:|---:|
| 50 | 15 | 3e-5 | 8e-5 | 0.001 |
| 100 | 20 | 2e-5 | 8e-5 | 0.001 |
| 200 | 20 | 1e-5 | 8e-5 | 0.001 |

**A fixed budget does not establish convergence.** Inspect loss trajectories and
initialization before attributing differences to model expressiveness.

## Covariance decomposition

For any T × N matrix X, let `Xc = X - column_mean(X)`. Sample covariance and the
fitted/residual cross term are:

```text
S(X) = Xc.T @ Xc / (T - 1)
C    = Fc.T @ Ec / (T - 1)
S(R) = S(F) + S(E) + C + C.T                 [exact identity]
Sigma_hat = S(F) + regularized_residual(E)   [before numerical repair]
```

The fitted estimator omits `C + C.T`; BCD does not enforce its disappearance.
Even using unregularized `S(E)` therefore need not recover `S(R)`. Diagnostics
record `||C||_F` as `fit_residual_cross_frobenius` (the norm of C, not C + C.T).

For full-sample covariance estimation without a rolling portfolio evaluation,
call `fit_model(returns, model_settings)` and pass its predictions and observed
returns to a registered estimator. This is an in-sample estimate.

## Covariance estimators

| Registry name | Construction |
|---|---|
| `sample` | `S(R) + ridge * I`; default ridge `1e-6` |
| `linear_shrinkage` | Scikit-learn Ledoit–Wolf on observed returns |
| `fitted_residual` | `S(F)` plus adaptively thresholded residual covariance |
| `fitted_residual_regularized` | `S(F)` plus the residual treatment selected by `params.method` |

Sample covariances use `ddof=1`; Ledoit–Wolf uses scikit-learn's own normalization
and estimated shrinkage. All estimators apply a final positive-definiteness floor
(default `f = 1e-10`): symmetrize A, then add
`max(0, f - smallest_eigenvalue(A)) * I`.

### Adaptive residual thresholding

For `S = S(E)`, define the variability statistic and threshold scale:

```text
theta[i,j] = mean_t(abs(E[t,i] * E[t,j] - S[i,j]))
omega      = c_star * sqrt(T**(-beta/(beta+d)) * log(T)**8)
H[i,j]     = S[i,j] if abs(S[i,j]) >= theta[i,j] * omega, else 0
```

Defaults are `c_star=1`, `beta=2`, `d=3`; log is natural. The products in theta
use uncentered residuals, although S is centered. Thresholding includes the
**diagonal**, so it can remove residual variances as well as cross-covariances.

If H has an eigenvalue below `eps=1e-8`, clip its eigenvalues to eps and add
`eps * I`. Otherwise leave H unchanged. Add this residual estimate to `S(F)`,
then apply the final floor above. If every residual entry is removed, the repair
produces `2 * eps * I`: the estimate is essentially fitted covariance plus a
small diagonal term. This does not imply that true residual risk is negligible.
Inspect sparsity, removed diagonal entries, eigenvalues, and repair magnitude.

### Other residual treatments

`fitted_residual_regularized` supports:

- `diagonal` (default): retain only the diagonal of `S(E)`.
- `sample`: use all of `S(E)`.
- `ledoit_wolf`: fit Ledoit–Wolf to E.
- `paper_diagonal`: use the same theta and threshold rule with
  `omega = 3 * sqrt(log(N)/T)` on off-diagonals. Retain sample variances, floored
  at f. Convert to correlation Q, then replace Q with `(1-a)*Q + a*I`, using the
  smallest a in [0,1] that raises its minimum eigenvalue to
  `min(1, f / min(floored_variances))`. Rescale to covariance.

The final covariance floor still applies. Threshold motivation is documented in
[Caner and Daniele's nonlinear-factor covariance work](https://arxiv.org/html/2209.04512v3).
Its neural-network theory does not automatically provide guarantees for BCD
residuals. Diagonal preservation and correlation repair are explicit estimator
choices here.

## Portfolio evaluation

Global minimum-variance weights solve `min_w w.T @ Sigma_hat @ w` subject to
`sum(w)=1`:

```text
v = solve(Sigma_hat, ones)
w = v / sum(v)
```

Equal weights `w[i]=1/N` are also available. Short positions are allowed;
transaction costs are omitted. Weights stay fixed when scoring each evaluation
window. With annual risk-free rate r and D trading days:

```text
daily_risk_free = (1 + r)**(1/D) - 1
excess[t]      = log_returns[t] @ w - daily_risk_free
AV             = D * mean(excess) * 100
SD             = sqrt(D) * std(excess, ddof=1) * 100
SR             = AV / SD
```

Defaults: `r=0.04`, `D=250`. AV and SD are annualized percentage-point scores;
SR is the Sharpe ratio (unavailable at zero volatility). These use a weighted
log-return score with a simple daily risk-free subtraction, **not an exact
self-financing simple-return wealth process**.

Daily scores are concatenated chronologically across nonoverlapping evaluation
windows before computing metrics; window Sharpes are not averaged. Reports
require complete, unique window/method records and aligned, nonduplicated dates.
They regenerate metrics from saved scores without refitting. Portfolio scores
alone do not establish covariance accuracy or optimizer convergence.
