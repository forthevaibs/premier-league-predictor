"""JSON subprocess endpoint: load model/history once for a batch of fixtures."""
import copy
import json
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from predictor import ROOT, History, load_matches, LABELS


def forecast_batch(request):
    as_of = pd.Timestamp(request['as_of'])
    if as_of.tzinfo is None:
        raise ValueError('as_of must include a timezone')
    cutoff = as_of.tz_convert('UTC').tz_localize(None).normalize()
    bundle = joblib.load(ROOT / 'artifacts/model.joblib')
    if bundle['trained_through'] >= cutoff:
        raise ValueError('Model training cutoff is not before forecast time; historical use would leak outcomes.')
    if 'poisson' not in bundle:
        raise ValueError('Train the upgraded match model first.')
    matches = load_matches()
    matches = matches[matches.Date < cutoff]
    if matches.empty:
        raise ValueError('No results before forecast time')
    history = History()
    for row in matches.itertuples():
        history.update(row)
    known = set(matches.HomeTeam) | set(matches.AwayTeam)
    rows, accepted, errors = [], [], {}
    for fixture in request['fixtures']:
        key = str(fixture['id'])
        kickoff = pd.Timestamp(fixture['kickoff_time'])
        home, away = fixture['home'], fixture['away']
        if kickoff.tzinfo is None or kickoff < as_of:
            errors[key] = 'Fixture has started or kickoff timezone is missing'; continue
        if home == away or home not in known or away not in known:
            errors[key] = f'Unmapped team: {home} / {away}'; continue
        # Each fixture is a scenario based on today's known results. Future fixtures
        # never update history or manufacture outcomes for subsequent gameweeks.
        state = copy.deepcopy(history)
        date = kickoff.tz_convert('UTC').tz_localize(None).normalize()
        state.prepare(home, away, date.year-int(date.month < 7))
        rows.append(state.features(home, away, date))
        accepted.append(fixture)
    results = {}
    if rows:
        x = pd.DataFrame(rows)[bundle['columns']]
        rates = bundle['poisson'].predict_rates(x)
        probabilities = bundle['model'].predict_proba(x)
        for f, r, p in zip(accepted, rates, probabilities):
            results[str(f['id'])] = dict(home=f['home'], away=f['away'], kickoff_time=f['kickoff_time'],
                home_goals=float(r[0]), away_goals=float(r[1]),
                home_clean_sheet=float(np.exp(-r[1])), away_clean_sheet=float(np.exp(-r[0])),
                probabilities=dict(zip(LABELS, p.tolist())))
    return dict(fixtures=results, errors=errors, model=bundle['selected'],
        trained_through=str(bundle['trained_through'].date()), data_through=str(matches.Date.max().date()),
        data_age_days=int((cutoff-matches.Date.max()).days), as_of=as_of.isoformat())


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        print(json.dumps(forecast_batch(json.load(sys.stdin))))
