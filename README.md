# SPY Signal Robustness Study & Viewer

[![tests](https://github.com/Manan-KK/signal-robustness-study/actions/workflows/test.yml/badge.svg)](https://github.com/Manan-KK/signal-robustness-study/actions/workflows/test.yml)

A small, auditable research package for answering a narrow question: does a
market-event model retain skill when every forecast is trained only on labels
that were fully resolved before the decision date?

This public edition contains **no SPY prices or other market data**. Its runnable
demo uses a seeded synthetic price series, and its analysis command accepts a
caller-supplied `date, close` CSV. The browser viewer is generated from aggregate
scores only and has no JavaScript dependency.

## What this repository demonstrates

- Backward-looking price features with an explicit decision timestamp.
- Forward labels that are unavailable for training until their horizon ends.
- Five models evaluated on the same rows and the same resolved training sets.
- Anchored walk-forward splits plus purged, embargoed combinatorial splits.
- A parameter surface that reports every tested window pair, not just the best.
- Brier skill and ROC AUC alongside a history-only base-rate forecast.
- A self-contained HTML viewer generated from aggregate sweep results.

“Time-ordered” here means that a training label is admitted only when
`outcome_end_date < decision_date`. It does **not** mean causal inference and it
does not identify the effect of a trading rule.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
signal-robustness demo --output-dir demo_output
```

Open `demo_output/viewer.html` after the demo completes. The command also writes
aggregate model, sweep, and split-audit CSVs. `demo_output/` is ignored by Git.

To evaluate data you are permitted to use:

```bash
signal-robustness analyze \
  --prices /path/to/prices.csv \
  --output-dir local_output
```

The input must contain increasing, unique dates and finite positive closes. The
tool does not download, copy, or redistribute market data.

## Public synthetic demo versus private-run evidence

The synthetic demo verifies software behavior. It is not economic evidence and
its scores should not be compared with a real-market study.

The earlier private research run produced the following **preserved evidence,
not reproduced by this public repository**:

- 286 parameter variants across 22 anchored walk-forward folds and 45 purged
  combinatorial splits.
- Five models evaluated on identical labels, with up to 1,101 observations per
  model for the principal 20-trading-day half-recovery definition.
- All five models had negative Brier skill versus the history-only base-rate
  forecast for that definition. A different full-recovery label showed limited,
  model-dependent positive skill, so the honest finding was weak and
  label-sensitive—not a durable edge.

Those figures remain provenance-labeled private-run evidence until a user with
rights to the underlying data reproduces them through a public interface. No
private forecast rows, manifests, source hashes, or derived market series are
included here.

## Models in the public demo

The public edition implements five deliberately compact probability benchmarks:

1. beta-smoothed base rate;
2. fixed quantile bin;
3. nearest analog;
4. ridge logistic regression; and
5. Gaussian naive Bayes.

They are implemented directly with NumPy so the training cutoff and probability
calculation remain inspectable. They are educational reference implementations,
not optimized estimators.

## Outputs

| File | Contents |
|---|---|
| `model_summary.csv` | Same-label observation count, Brier score, history-base-rate Brier score, Brier skill, and ROC AUC. |
| `window_sweep.csv` | One row for every declared fast/slow window pair. |
| `split_audit.csv` | Train/test sizes, purged rows, embargoed rows, and overlap checks. |
| `viewer.html` | Dependency-free interactive table generated from aggregate sweep scores. |

No output records the input path or embeds raw observations.

## Limitations

- The included demo is synthetic and supports no investment conclusion.
- A forward label necessarily uses future prices for scoring; the contract only
  prevents unresolved labels from entering training.
- Combinatorial splits are a stability diagnostic, not a simulation of what a
  live forecaster knew through time.
- Parameter sweeps create multiple-comparison risk. The viewer exposes the full
  surface but does not supply a statistical correction.
- The study omits execution, costs, liquidity, taxes, and portfolio accounting.
- Brier skill is relative to a rolling history-only base rate and can be unstable
  in small samples.
- The lightweight models prioritize auditability over numerical optimization.

## Development history

This repository was extracted as a clean, independently implemented public
edition after an external-sharing review. The public commit history begins with
the sanitized release and therefore does not represent the full period of the
earlier private research. No client information, employer code, licensed
datasets, internal reports, or firm-specific strategy thresholds are included.

## License and dependencies

Project code is released under the [MIT License](LICENSE). Runtime dependencies
are NumPy and pandas; neither is vendored. The viewer uses browser APIs only.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This repository is research software, not investment advice.
