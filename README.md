# Premier League match predictor

Pre-match home/draw/away probabilities using pandas, scikit-learn and XGBoost, with Poisson goal models and probability blends. Linear regression remains a separate forecast for goals, shots and shots on target. A Streamlit interface and command-line interface use the same feature pipeline.

## Second-milestone measured results

The three expanding validation windows selected a 75% Poisson / 25% XGBoost probability blend. Search evaluated 30 standalone candidates and 6 blends. The selected XGBoost component uses depth 2, learning rate .02, 250 rounds, minimum child weight 30, L2 penalty 15, row subsampling .75 and column subsampling .7. The Poisson component uses alpha=1.

| Model | Benchmark accuracy | Benchmark log loss |
| --- | ---: | ---: |
| Previous deployed logistic model | 49.1% | 1.0206 |
| Enhanced logistic model | 50.1% | 1.0097 |
| Tuned XGBoost | 51.1% | 1.0164 |
| Poisson model | 51.3% | 1.0005 |
| Validation-selected Poisson/XGBoost blend | 50.9% | 1.0030 |

All comparisons use the same 760-match benchmark. Selection followed validation log loss, so the deployed blend is not necessarily the best model on this benchmark. The increase over the prior deployed model is 14 correct outcomes (1.84 percentage points); it is not evidence of a statistically established improvement. Draw recall remains 0% for the selected blend: it emits draw probabilities but did not rank draw highest in any benchmark match. This limitation is displayed through per-outcome recall in the app. The period was previously inspected; a fresh prospective evaluation is still needed.

All 18 unit/integration tests pass, covering temporal leakage, ratings, Poisson scoreline equivalence, serialization, blend arithmetic and the Streamlit prediction form.

## First-milestone reference results

Trained on 2,660 matches, selected on 380 validation matches, tested on 760 matches. The 83-feature logistic model was selected using validation log loss.

| Model | Test accuracy | Test log loss | Macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic regression (selected) | 49.1% | 1.0206 | 0.3727 |
| XGBoost depth 2 | 50.8% | 1.0228 | 0.3831 |
| XGBoost depth 3 | 50.4% | 1.0263 | 0.3791 |
| Historical frequencies | 41.7% | 1.0837 | 0.1962 |

The modest macro-F1 indicates uneven performance across outcomes; this is an initial benchmark, not a high-confidence winner predictor. Home/away goals MAE was 0.98/0.89. Data ends on 24 May 2026. Refresh current-season data before treating outputs as current forecasts.

## Run on Windows PowerShell

```powershell
git clone https://github.com/forthevaibs/premier-league-predictor.git
cd premier-league-predictor
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python predictor.py download
.\.venv\Scripts\python predictor.py train
.\.venv\Scripts\python -m streamlit run app.py
```

The prepared environment in this workspace inherits existing system packages; the commands above also support a clean environment. Data downloads cache ten seasons, 2016/17–2025/26, by default. Source URLs and SHA-256 checksums are recorded in `data/sources.json`. Use `download --start 2026 --end 2026 --refresh` to fetch or refresh the current season, then retrain.

```powershell
.\.venv\Scripts\python predictor.py predict --home Arsenal --away Chelsea --date 2026-10-19
.\.venv\Scripts\python -m unittest -v
```

Prediction dates must be after the model's training cutoff. The team names must match source names, e.g. `Man City`, `Man United`, `Tottenham`. All historical teams are selectable to permit hypothetical fixtures; selection is not a check of current league membership. Historical held-out predictions are saved separately.

## Data and permitted use

Source: https://football-data.co.uk/englandm.php
Dictionary: https://football-data.co.uk/notes.txt
Usage statement: https://football-data.co.uk/data.php

The provider describes the data as intended for private individuals and excludes commercial/data-training products using automated bots/scrapers/AI. This is a local personal prototype; do not assume the data is openly licensed for redistribution or commercial deployment. Obtain suitable permission or substitute licensed CSVs before those uses. GitHub mirrors do not change upstream usage terms. Data is downloaded directly, not committed to source.

## Features and leakage protection

Each row is one match with separate home/away values and differences. Features include last-five and last-20 averages for points, goals scored/conceded, shots and shots on target for/against, clean sheets, last-five venue-specific averages, rest days, history count, Elo, and head-to-head points/count over up to five meetings in three years. Team form excludes matches older than 365 days. Elo uses K=20 and a 65-point home adjustment, fixed before evaluation.

The second milestone adds:

- Time-weighted form over the last 20 matches, with a 60-day half-life.
- Opponent-adjusted result: actual result (1/.5/0) minus the pre-match Elo expected result. A win against a stronger opponent yields a larger positive residual. Opponent Elo is captured before the result, never read retroactively from its latest value.
- Separate online log-scale attack and defence ratings. Expected home/away goals start at 1.5/1.2 times exp(own attack minus opposing defence); updates use goal residuals with learning rate .03. Positive defence ratings mean stronger defence. Expected rates are bounded during online updates for stability. Weighted attack/defence residuals are also features.
- At each new season, 25% of Elo deviation from 1500 and attack/defence ratings regress toward neutral. Teams absent from the immediately preceding observed Premier League season start at Elo 1400 and attack/defence -0.1, including returning clubs. The first dataset season starts all teams at neutral. These are fixed priors, not lower-league estimates or parameters chosen from the benchmark. A July boundary is used for future dates; historical rows use source season labels, including the COVID-delayed season.

All fixtures on a date are featurized before any results that date update history. Neither current-match stats, odds, final league tables nor team identities enter the model. Missing values are imputed using training data with missingness indicators; scaling is fitted on training data. Cup/European matches, injuries, player matchups and possession are outside this milestone. Rest means league rest. Rolling form spans season boundaries within the one-year limit.

## Evaluation

- Expanding validation: train on seasons before 2021, predict 2021/22; train before 2022, predict 2022/23; train before 2023, predict 2023/24. Each preprocessing pipeline is fit separately within its training window.
- Search: 3 logistic regularization values, 3 Poisson regularization values and 24 reproducible random XGBoost configurations. Search covers depth 2/3/4, learning rates .02/.05/.1, rounds 120/250/450, child weight, L1/L2 penalties and row/column sampling. This is a bounded randomized search, not an exhaustive grid. Configurations and per-window metrics are saved.
- Poisson: separate regularized home/away goal regressions. Independent Poisson scoreline probabilities are summed exactly using the Skellam goal-difference distribution, including all tails. Independence is a simplifying assumption; low-score correlation is not explicitly modelled.
- Blends: best logistic and best XGBoost are each blended with the best Poisson model at Poisson weights .25/.5/.75. Selection uses pooled out-of-fold validation log loss, never benchmark scores.
- Benchmark: refit family winners on seasons through 2023/24; compare models/blends on 2024/25–2025/26. Earlier benchmark results update form/Elo; weights remain fixed. This period was already inspected in milestone one and is a reused benchmark, not a fresh untouched holdout. New-season prospective validation is needed before claiming generalization gains.
- Deployment: refit the validation-selected model, best Poisson model and linear regressions on all available completed matches. Deployment is separate from evaluation. Additional current seasons affect deployment, not the fixed benchmark.

`artifacts/evaluation.json` contains search configurations, fold metrics, accuracy, macro-F1, log loss, multiclass Brier score, per-class reliability bins, confusion matrices, recall, statistics MAE and split dates. `artifacts/validation_leaderboard.csv` ranks all candidates. `artifacts/test_predictions.csv` contains benchmark probabilities. Previous model/report/predictions are copied to timestamped `artifacts/previous_runs/` folders before replacement. Historical-frequency and always-home baselines provide context. No post-hoc calibrator is fitted. Regression forecasts are clipped to zero; displayed shots-on-target are additionally bounded by shots. Regression MAE is measured after nonnegative clipping only.

`artifacts/model.joblib` is a local trusted artifact; never load untrusted joblib files. Install project dependencies from `requirements.txt`. Raw data, trained artifacts, machine-specific package inventories and secrets are excluded from Git. Run download and train locally to generate the model before launching the app. CI runs the data-independent tests; app integration tests are skipped until a trained artifact exists.

## Optional FPL integration

`fixture_batch.py` exposes a JSON stdin/stdout endpoint for the separate FPL advisor. Its input includes a timezone-aware `as_of` timestamp and fixtures with `id`, `home`, `away` and `kickoff_time`. It returns expected goals, clean-sheet probabilities, outcome probabilities and data/training cutoffs. The advisor and its comparison records are separate from this repository. For the existing advisor's default folder lookup, clone this repository into a sibling folder named `match_predictor`, or change its `fixture_bridge.py` `MATCH` setting.

The default completed-season dataset is not a live form feed. The app displays the data cutoff and warns for stale histories. To add current results, download the current season explicitly and retrain. Refresh cached current-season files when updating.
