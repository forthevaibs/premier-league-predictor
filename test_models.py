import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import joblib
from scipy.stats import poisson
from models import poisson_probabilities, ProbabilityBlend, PoissonOutcome
from predictor import History, feature_table
from test_predictor import fixtures
from training import folds, candidate_specs


class ImprovedModelTests(unittest.TestCase):
    def test_poisson_matches_scoreline_sum(self):
        h, a = 1.8, 1.1
        matrix = np.outer(poisson.pmf(np.arange(40), h), poisson.pmf(np.arange(40), a))
        expected = [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
        np.testing.assert_allclose(poisson_probabilities([h], [a])[0], expected, atol=1e-12)

    def test_poisson_symmetry_and_extremes(self):
        p = poisson_probabilities([.01, 1.5, 30], [.01, 1.5, .1])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(p[1, 0], p[1, 2])
        self.assertGreater(p[0, 1], .98)
        self.assertGreater(p[2, 0], .99)

    def test_season_shrink_and_promoted_prior(self):
        history = History()
        history.prepare('A', 'B', 2020)
        history.elo['A'] = 1700
        history.prepare('A', 'C', 2021)
        self.assertEqual(history.elo['A'], 1650)
        self.assertEqual(history.elo['C'], 1400)
        history.prepare('A', 'C', 2021)
        self.assertEqual(history.elo['A'], 1650)
        history.prepare('A', 'B', 2022)
        self.assertEqual(history.elo['B'], 1400)

    def test_opponent_adjustment_uses_prior_strength(self):
        row = next(fixtures().itertuples())
        weak, strong = History(), History()
        weak.elo['B'], strong.elo['B'] = 1300, 1700
        weak.update(row)
        strong.update(row)
        self.assertGreater(strong.games['A'][0]['adjusted_result'], weak.games['A'][0]['adjusted_result'])
        self.assertEqual(strong.games['A'][0]['opponent_elo'], 1700)

    def test_recency_weighting(self):
        frame = fixtures().iloc[[0, 2]].copy()
        frame.loc[2, 'Date'] = pd.Timestamp('2024-03-01')
        history = History()
        for row in frame.itertuples():
            history.update(row)
        f = history.team_features('A', 'H', pd.Timestamp('2024-03-02'))
        self.assertGreater(f['gf_weighted'], f['gf_last5'])
        self.assertGreater(f['attack_rating'], 0)

    def test_validation_excludes_benchmark(self):
        frame = pd.DataFrame({'Season': range(2016, 2026), 'Date': pd.to_datetime([f'{year}-08-01' for year in range(2016, 2026)])})
        for _, before, val in folds(frame):
            self.assertLess(frame.loc[before, 'Date'].max(), frame.loc[val, 'Date'].min())
            self.assertLess(frame.loc[val, 'Season'].max(), 2024)
        configs = [s for s in candidate_specs().values() if s['kind'] == 'xgboost']
        self.assertEqual(len(configs), 24)
        for param in ['max_depth', 'learning_rate', 'n_estimators', 'reg_lambda', 'reg_alpha']:
            self.assertGreater(len({s[param] for s in configs}), 1)

    def test_blend_is_weighted_probability(self):
        class Constant:
            def __init__(self, p): self.p = np.array([p])
            def predict_proba(self, x): return self.p
        model = ProbabilityBlend(Constant([.6, .2, .2]), Constant([.2, .4, .4]), .25)
        np.testing.assert_allclose(model.predict_proba(None), [[.5, .25, .25]])

    def test_poisson_fit_positive_rates(self):
        x = pd.DataFrame({'strength': [-1., 0., 1., 2., 3., 4.]})
        goals = pd.DataFrame({'FTHG': [0, 1, 2, 1, 3, 4], 'FTAG': [2, 1, 1, 0, 1, 0]})
        model = PoissonOutcome().fit(x, goals)
        self.assertTrue((model.predict_rates(x) > 0).all())
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poisson.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(x), model.predict_proba(x))


if __name__ == '__main__':
    unittest.main()
