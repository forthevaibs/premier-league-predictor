import unittest
from streamlit.testing.v1 import AppTest
from predictor import ROOT, predict


@unittest.skipUnless((ROOT / 'artifacts/model.joblib').exists(), 'Run training first')
class AppTests(unittest.TestCase):
    def test_prediction_form(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)

    def test_same_team_rejected(self):
        with self.assertRaises(ValueError):
            predict('Arsenal', 'Arsenal', '2030-01-01')

    def test_historical_date_rejected(self):
        with self.assertRaises(ValueError):
            predict('Arsenal', 'Chelsea', '2020-01-01')

    def test_probability_sum(self):
        result = predict('Arsenal', 'Chelsea', '2030-01-01')
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in result['probabilities'].values()))


if __name__ == '__main__':
    unittest.main()
