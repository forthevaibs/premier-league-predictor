"""Expanding-window model search. Test outcomes never choose model parameters."""
from datetime import datetime, timezone
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, confusion_matrix, recall_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from models import PoissonOutcome, ProbabilityBlend
from predictor import ROOT, TARGETS, LABELS, load_matches, feature_table, pipeline, score


def candidate_specs():
    specs = {f'logistic_C{c}': {'kind': 'logistic', 'C': c} for c in [.01, .1, 1]}
    specs.update({f'poisson_alpha{a}': {'kind': 'poisson', 'alpha': a} for a in [.1, 1, 10]})
    space = dict(max_depth=[2, 3, 4], learning_rate=[.02, .05, .1],
        n_estimators=[120, 250, 450], min_child_weight=[5, 15, 30],
        reg_lambda=[1, 5, 15], reg_alpha=[0, .5, 2],
        subsample=[.75, 1.0], colsample_bytree=[.7, 1.0])
    for i, params in enumerate(ParameterSampler(space, n_iter=24, random_state=42)):
        specs[f'xgb_{i:02}'] = {'kind': 'xgboost', **params}
    return specs


def build(spec):
    params = dict(spec)
    kind = params.pop('kind')
    if kind == 'poisson':
        return PoissonOutcome(**params)
    if kind == 'logistic':
        return pipeline(LogisticRegression(**params, max_iter=2000))
    return pipeline(XGBClassifier(**params, objective='multi:softprob', num_class=3,
        eval_metric='mlogloss', random_state=42, n_jobs=2, tree_method='hist'))


def fit(spec, x, matches):
    model = build(spec)
    return model.fit(x, matches[['FTHG', 'FTAG']] if spec['kind'] == 'poisson' else matches.outcome)


def folds(matches):
    result = []
    for season in [2021, 2022, 2023]:
        before, validation = matches.Season < season, matches.Season == season
        if not before.any() or not validation.any():
            raise ValueError(f'Missing training or validation data for season {season}')
        if matches.loc[before, 'Date'].max() >= matches.loc[validation, 'Date'].min():
            raise ValueError('Season folds are not chronological')
        result.append((season, before, validation))
    return result


def detailed_score(y, probabilities):
    metrics = score(y, probabilities)
    pred = probabilities.argmax(axis=1)
    metrics['confusion_matrix'] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    metrics['recall_by_outcome'] = dict(zip(LABELS, recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist()))
    return metrics


def run_training():
    matches = load_matches()
    print(f'Building enhanced features for {len(matches)} matches...', flush=True)
    x = feature_table(matches)
    specs = candidate_specs()
    windows = folds(matches)
    report = {'schema_version': 2, 'features': len(x.columns), 'candidate_specs': specs,
        'selection_metric': 'pooled expanding-window validation log loss',
        'test_note': '2024/25-2025/26 is a reused benchmark, not a fresh untouched holdout.',
        'validation_windows': [], 'validation': {}, 'test': {}}
    for year, before, val in windows:
        report['validation_windows'].append(dict(season=year, train_matches=int(before.sum()), validation_matches=int(val.sum()),
            train_end=str(matches.loc[before, 'Date'].max().date()), validation_start=str(matches.loc[val, 'Date'].min().date())))
    out_of_fold = {}
    targets = pd.concat([matches.loc[val, 'outcome'] for _, _, val in windows])
    for number, (name, spec) in enumerate(specs.items(), 1):
        predictions, per_fold = [], []
        for year, before, val in windows:
            model = fit(spec, x[before], matches[before])
            probabilities = model.predict_proba(x[val])
            predictions.append(probabilities)
            per_fold.append(dict(season=year, **{k: v for k, v in score(matches.loc[val, 'outcome'], probabilities).items() if k != 'calibration'}))
        out_of_fold[name] = np.vstack(predictions)
        report['validation'][name] = detailed_score(targets, out_of_fold[name])
        report['validation'][name]['folds'] = per_fold
        print(f'[{number}/{len(specs)}] {name}: validation loss {report["validation"][name]["log_loss"]:.4f}', flush=True)
    best = {}
    for kind in ['logistic', 'xgboost', 'poisson']:
        best[kind] = min([name for name, spec in specs.items() if spec['kind'] == kind], key=lambda name: report['validation'][name]['log_loss'])
    blend_specs = {}
    for classifier in [best['logistic'], best['xgboost']]:
        for weight in [.25, .5, .75]:
            name = f'blend_{classifier}_poisson_{weight}'
            blend_specs[name] = dict(classifier=classifier, poisson=best['poisson'], weight=weight)
            out_of_fold[name] = (1-weight)*out_of_fold[classifier] + weight*out_of_fold[best['poisson']]
            report['validation'][name] = detailed_score(targets, out_of_fold[name])
    report['blend_specs'] = blend_specs
    selected = min(report['validation'], key=lambda name: report['validation'][name]['log_loss'])
    report['selected_model'] = selected
    report['best_by_family'] = best
    development, test = matches.Season <= 2023, matches.Season.isin([2024, 2025])
    if not test.any():
        raise ValueError('Need benchmark seasons 2024 and/or 2025')
    report['split'] = {name: dict(matches=int(mask.sum()), start=str(matches.loc[mask, 'Date'].min().date()), end=str(matches.loc[mask, 'Date'].max().date()))
                       for name, mask in [('development', development), ('test', test)]}
    fitted = {name: fit(specs[name], x[development], matches[development]) for name in best.values()}
    for name, spec in blend_specs.items():
        fitted[name] = ProbabilityBlend(fitted[spec['classifier']], fitted[spec['poisson']], spec['weight'])
    for name, model in fitted.items():
        report['test'][name] = detailed_score(matches.loc[test, 'outcome'], model.predict_proba(x[test]))
    freq = np.bincount(matches.loc[development, 'outcome'], minlength=3) / development.sum()
    report['test']['historical_frequencies'] = detailed_score(matches.loc[test, 'outcome'], np.tile(freq, (test.sum(), 1)))
    report['always_home_accuracy'] = float((matches.loc[test, 'outcome'] == 0).mean())
    regression = pipeline(LinearRegression())
    complete = matches[TARGETS].notna().all(axis=1)
    regression.fit(x[development & complete], matches.loc[development & complete, TARGETS])
    report['stats_mae'] = dict(zip(TARGETS, mean_absolute_error(matches.loc[test & complete, TARGETS],
        np.maximum(regression.predict(x[test & complete]), 0), multioutput='raw_values').tolist()))
    folder = ROOT / 'artifacts'
    folder.mkdir(exist_ok=True)
    archive = folder / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    archive.mkdir(parents=True)
    for name in ['evaluation.json', 'test_predictions.csv', 'model.joblib']:
        if (folder / name).exists():
            shutil.copy2(folder / name, archive / name)
    if (folder / 'evaluation.json').exists():
        previous = json.loads((folder / 'evaluation.json').read_text())
        report['previous_selected_test'] = dict(model=previous['selected_model'], **previous['test'][previous['selected_model']])
    output = matches.loc[test, ['Date', 'HomeTeam', 'AwayTeam', 'outcome']].copy()
    output[LABELS] = fitted[selected].predict_proba(x[test])
    output.to_csv(folder / 'test_predictions.csv', index=False)
    print(f'Selected {selected}; refitting deployment model...', flush=True)
    if selected in blend_specs:
        blend = blend_specs[selected]
        model = ProbabilityBlend(fit(specs[blend['classifier']], x, matches), fit(specs[blend['poisson']], x, matches), blend['weight'])
    else:
        model = fit(specs[selected], x, matches)
    poisson = fit(specs[best['poisson']], x, matches)
    regression.fit(x[complete], matches.loc[complete, TARGETS])
    # Serialize to a temporary file and replace after fitting succeeds.
    temporary = folder / 'model.tmp'
    joblib.dump(dict(model=model, poisson=poisson, regression=regression, columns=list(x.columns),
        trained_through=matches.Date.max(), selected=selected, feature_version=2), temporary)
    temporary.replace(folder / 'model.joblib')
    (folder / 'evaluation.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame({name: {k: v for k, v in metrics.items() if k in ['accuracy', 'macro_f1', 'log_loss', 'brier_multiclass']}
        for name, metrics in report['validation'].items()}).T.sort_values('log_loss').to_csv(folder / 'validation_leaderboard.csv')
    print(json.dumps({'selected': selected, 'test': report['test'][selected]}, indent=2), flush=True)
    return report
