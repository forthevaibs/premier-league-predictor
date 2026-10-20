import unittest
import numpy as np
import pandas as pd
from predictor import History, feature_table, score


def fixtures():
    rows = [
        ['2024-01-01', 'A', 'B', 2, 0],
        ['2024-01-01', 'C', 'D', 1, 1],
        ['2024-01-08', 'B', 'A', 1, 3],
        ['2024-01-15', 'A', 'C', 0, 0],
    ]
    frame = pd.DataFrame(rows, columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    frame.Date = pd.to_datetime(frame.Date)
    for col in ['HS', 'AS', 'HST', 'AST']:
        frame[col] = 4
    return frame


class FeatureTests(unittest.TestCase):
    def test_current_and_future_results_do_not_change_features(self):
        original = fixtures()
        altered = original.copy()
        altered.loc[2:, ['FTHG', 'FTAG', 'HS', 'AST']] = 99
        pd.testing.assert_frame_equal(feature_table(original).iloc[:3], feature_table(altered).iloc[:3])

    def test_same_day_is_frozen(self):
        frame = fixtures()
        frame.loc[1, 'HomeTeam'] = 'A'
        features = feature_table(frame)
        self.assertEqual(features.loc[1, 'home_history_count'], 0)
        self.assertEqual(features.loc[1, 'home_elo'], 1500)

    def test_home_away_orientation_and_h2h(self):
        features = feature_table(fixtures())
        self.assertEqual(features.loc[2, 'home_gf_last5'], 0)
        self.assertEqual(features.loc[2, 'away_gf_last5'], 2)
        self.assertEqual(features.loc[2, 'h2h_home_points'], 0)
        self.assertEqual(features.loc[2, 'home_rest_days'], 7)
        self.assertEqual(features.loc[3, 'home_gf_last5'], 2.5)

    def test_future_feature_path_matches_training(self):
        frame = fixtures()
        history = History()
        for row in frame.iloc[:2].itertuples():
            history.update(row)
        expected = history.features('B', 'A', pd.Timestamp('2024-01-08'))
        pd.testing.assert_series_equal(pd.Series(expected), feature_table(frame).iloc[2], check_names=False)

    def test_cold_start_and_old_history(self):
        history = History()
        for row in fixtures().itertuples():
            history.update(row)
        features = history.features('A', 'New', pd.Timestamp('2026-01-01'))
        self.assertEqual(features['home_history_count'], 0)
        self.assertTrue(np.isnan(features['away_points_last5']))

    def test_probabilities_and_metrics(self):
        metrics = score(pd.Series([0, 1, 2]), np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]))
        self.assertEqual(metrics['accuracy'], 1)
        self.assertLess(metrics['log_loss'], .3)


if __name__ == '__main__':
    unittest.main()
