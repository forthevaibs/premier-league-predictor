"""Premier League predictor. All features are computed before a match date."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LABELS = ['Home win', 'Draw', 'Away win']
STATS = ['gf', 'ga', 'shots', 'shots_against', 'sot', 'sot_against', 'points', 'clean_sheet']
TARGETS = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']


def download(start=2016, end=2025, refresh=False):
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    manifest_path = folder / 'sources.json'
    manifest = {item['season']: item for item in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    for year in range(start, end + 1):
        code = f'{year % 100:02}{(year + 1) % 100:02}'
        url = f'https://www.football-data.co.uk/mmz4281/{code}/E0.csv'
        path = folder / f'{year}.csv'
        if refresh or not path.exists():
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            if b'HomeTeam' not in payload or b'FTHG' not in payload:
                raise ValueError(f'Invalid CSV response: {url}')
            path.write_bytes(payload)
        manifest[year] = dict(season=year, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(f'Cached {year}/{year+1}', flush=True)
    manifest_path.write_text(json.dumps(list(manifest.values()), indent=2))


def load_matches():
    frames = []
    for path in sorted((ROOT / 'data').glob('[0-9][0-9][0-9][0-9].csv')):
        frame = pd.read_csv(path)
        required = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
        if not required.issubset(frame):
            raise ValueError(f'Missing required columns: {path}')
        frame = frame.dropna(subset=list(required)).copy()
        frame['Date'] = pd.to_datetime(frame['Date'], dayfirst=True, format='mixed', errors='raise').dt.normalize()
        frame['Season'] = int(path.stem)
        for col in TARGETS:
            frame[col] = pd.to_numeric(frame[col], errors='raise') if col in frame else np.nan
        for col in ['HomeTeam', 'AwayTeam']:
            frame[col] = frame[col].str.strip()
        if (frame[['FTHG', 'FTAG']] < 0).any().any():
            raise ValueError('Negative score')
        frames.append(frame)
    if not frames:
        raise ValueError('No data. Run: python predictor.py download')
    matches = pd.concat(frames, ignore_index=True).sort_values(['Date', 'HomeTeam']).reset_index(drop=True)
    if matches.duplicated(['Date', 'HomeTeam', 'AwayTeam']).any():
        raise ValueError('Duplicate fixtures detected')
    matches['outcome'] = np.where(matches.FTHG > matches.FTAG, 0, np.where(matches.FTHG == matches.FTAG, 1, 2))
    return matches


class History:
    def __init__(self):
        self.games = defaultdict(list)
        self.elo = defaultdict(lambda: 1500.0)
        self.attack = defaultdict(float)
        self.defence = defaultdict(float)
        self.season = None
        self.members = set()
        self.previous_members = set()

    def prepare(self, home, away, season):
        if self.season is not None and season < self.season:
            raise ValueError('History must be processed chronologically')
        if season != self.season:
            self.previous_members = self.members.copy()
            self.members = set()
            for team in self.previous_members:
                self.elo[team] = 1500 + .75 * (self.elo[team] - 1500)
                self.attack[team] *= .75
                self.defence[team] *= .75
            self.season = season
        for team in [home, away]:
            if team not in self.members:
                if self.previous_members and team not in self.previous_members:
                    # No lower-league data: a fixed promoted-team prior, including returnees.
                    self.elo[team] = 1400.0
                    self.attack[team] = -.1
                    self.defence[team] = -.1
                self.members.add(team)

    def team_features(self, team, venue, date):
        # Old-season histories expire; newly promoted/returning teams get missing indicators.
        games = [g for g in self.games[team] if (date - g['date']).days <= 365]
        result = {'elo': self.elo[team], 'history_count': len(games),
                  'attack_rating': self.attack[team], 'defence_rating': self.defence[team],
                  'rest_days': min((date-games[-1]['date']).days, 60) if games else np.nan}
        for name, subset in [('last5', games[-5:]), ('last20', games[-20:]),
                             ('venue5', [g for g in games if g['venue'] == venue][-5:])]:
            for stat in STATS:
                values = [g[stat] for g in subset if pd.notna(g[stat])]
                result[f'{stat}_{name}'] = float(np.mean(values)) if values else np.nan
        recent = games[-20:]
        # 60-day half-life; opponent strength is captured at the historical kickoff.
        for stat in STATS + ['adjusted_result', 'attack_residual', 'defence_residual', 'opponent_elo']:
            usable = [g for g in recent if pd.notna(g[stat])]
            weights = [2 ** (-(date-g['date']).days / 60) for g in usable]
            result[f'{stat}_weighted'] = float(np.average([g[stat] for g in usable], weights=weights)) if usable else np.nan
        return result

    def features(self, home, away, date):
        h, a = self.team_features(home, 'H', date), self.team_features(away, 'A', date)
        result = {f'home_{k}': v for k, v in h.items()}
        result.update({f'away_{k}': v for k, v in a.items()})
        result.update({f'diff_{k}': h[k]-a[k] for k in h})
        meetings = [g for g in self.games[home] if g['opponent'] == away and (date-g['date']).days <= 1095][-5:]
        result['h2h_count'] = len(meetings)
        result['h2h_home_points'] = np.mean([g['points'] for g in meetings]) if meetings else np.nan
        return result

    def update(self, row):
        home, away = row.HomeTeam, row.AwayTeam
        self.prepare(home, away, getattr(row, 'Season', row.Date.year - int(row.Date.month < 7)))
        home_elo, away_elo = self.elo[home], self.elo[away]
        expected_home_goals = np.exp(np.clip(np.log(1.5) + self.attack[home] - self.defence[away], -2, 2))
        expected_away_goals = np.exp(np.clip(np.log(1.2) + self.attack[away] - self.defence[home], -2, 2))
        home_residual, away_residual = row.FTHG-expected_home_goals, row.FTAG-expected_away_goals
        expected = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - 65) / 400))
        actual = 1 if row.FTHG > row.FTAG else .5 if row.FTHG == row.FTAG else 0
        change = 20 * (actual - expected)
        self.elo[home] += change
        self.elo[away] -= change
        self.attack[home] += .03 * home_residual
        self.defence[away] -= .03 * home_residual
        self.attack[away] += .03 * away_residual
        self.defence[home] -= .03 * away_residual
        for team, opponent, venue, gf, ga, shots, against, sot, sot_against in [
            (home, away, 'H', row.FTHG, row.FTAG, row.HS, row.AS, row.HST, row.AST),
            (away, home, 'A', row.FTAG, row.FTHG, row.AS, row.HS, row.AST, row.HST)]:
            self.games[team].append(dict(date=row.Date, opponent=opponent, venue=venue,
                opponent_elo=away_elo if venue == 'H' else home_elo,
                adjusted_result=(actual-expected) * (1 if venue == 'H' else -1),
                attack_residual=home_residual if venue == 'H' else away_residual,
                defence_residual=-away_residual if venue == 'H' else -home_residual,
                gf=gf, ga=ga, shots=shots, shots_against=against, sot=sot, sot_against=sot_against,
                points=3 if gf > ga else 1 if gf == ga else 0, clean_sheet=int(ga == 0)))


def feature_table(matches):
    history, records = History(), []
    for date, group in matches.groupby('Date', sort=True):
        for row in group.itertuples():
            history.prepare(row.HomeTeam, row.AwayTeam, getattr(row, 'Season', date.year - int(date.month < 7)))
        # Freeze state for the whole date, so same-day results cannot leak.
        for row in group.itertuples():
            records.append({'index': row.Index, **history.features(row.HomeTeam, row.AwayTeam, date)})
        for row in group.itertuples():
            history.update(row)
    return pd.DataFrame(records).set_index('index').reindex(matches.index)


def pipeline(estimator):
    return make_pipeline(SimpleImputer(add_indicator=True, keep_empty_features=True), StandardScaler(), estimator)


def score(y, probabilities):
    # Tree output is float32; normalize after conversion to avoid blend roundoff warnings.
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    prediction = np.argmax(probabilities, axis=1)
    one_hot = np.eye(3)[np.asarray(y)]
    calibration = []
    for cls, label in enumerate(LABELS):
        for low in np.arange(0, 1, .2):
            mask = (probabilities[:, cls] >= low) & (probabilities[:, cls] < (low+.2 if low < .8 else 1.000001))
            if mask.any():
                calibration.append(dict(outcome=label, bin_start=float(low), count=int(mask.sum()),
                    mean_probability=float(probabilities[mask, cls].mean()), observed_rate=float(one_hot[mask, cls].mean())))
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0)),
                log_loss=float(log_loss(y, probabilities, labels=[0, 1, 2])),
                brier_multiclass=float(np.mean(np.sum((probabilities-one_hot)**2, axis=1))), calibration=calibration)


def train():
    from training import run_training
    with threadpool_limits(limits=2):
        return run_training()


def predict(home, away, date):
    bundle = joblib.load(ROOT / 'artifacts/model.joblib')
    date = pd.Timestamp(date).normalize()
    if date <= bundle['trained_through']:
        raise ValueError(f'Use a date after training cutoff {bundle["trained_through"].date()}; historical results are in test_predictions.csv.')
    matches = load_matches()
    known = set(matches.HomeTeam) | set(matches.AwayTeam)
    if home == away or home not in known or away not in known:
        raise ValueError('Choose two different teams present in the dataset.')
    history = History()
    for row in matches[matches.Date < date].itertuples():
        history.update(row)
    history.prepare(home, away, date.year - int(date.month < 7))
    features = history.features(home, away, date)
    x = pd.DataFrame([features])[bundle['columns']]
    probabilities = bundle['model'].predict_proba(x)[0]
    stats = np.maximum(bundle['regression'].predict(x)[0], 0)
    # Enforce physical consistency for the displayed shot forecasts.
    stats[4], stats[5] = min(stats[4], stats[2]), min(stats[5], stats[3])
    poisson = bundle.get('poisson')
    goal_model = dict(probabilities=dict(zip(LABELS, poisson.predict_proba(x)[0].tolist())),
                      goal_rates=poisson.predict_rates(x)[0].tolist()) if poisson is not None else None
    return dict(home=home, away=away, date=str(date.date()), model=bundle['selected'],
        trained_through=str(bundle['trained_through'].date()), data_through=str(matches.Date.max().date()),
        probabilities=dict(zip(LABELS, probabilities.tolist())), predicted_stats=dict(zip(TARGETS, stats.tolist())),
        poisson=goal_model,
        recent_matches={home: features['home_history_count'], away: features['away_history_count']},
        data_age_days=int((date-matches.Date.max()).days))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    dl = commands.add_parser('download')
    dl.add_argument('--start', type=int, default=2016)
    dl.add_argument('--end', type=int, default=2025)
    dl.add_argument('--refresh', action='store_true', help='Replace cached seasons with current source files')
    commands.add_parser('train')
    pred = commands.add_parser('predict')
    pred.add_argument('--home', required=True)
    pred.add_argument('--away', required=True)
    pred.add_argument('--date', required=True)
    args = parser.parse_args()
    if args.command == 'download':
        download(args.start, args.end, args.refresh)
    elif args.command == 'train':
        train()
    else:
        print(json.dumps(predict(args.home, args.away, args.date), indent=2))
