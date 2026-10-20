import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from fixture_batch import forecast_batch
from predictor import feature_table
from test_predictor import fixtures


class BatchTests(unittest.TestCase):
    def test_batch_loads_and_predicts_once_and_excludes_future_results(self):
        matches=fixtures()
        columns=list(feature_table(matches).columns)
        model=Mock(); model.predict_proba.return_value=np.array([[.5,.3,.2],[.5,.3,.2]])
        poisson=Mock(); poisson.predict_rates.return_value=np.array([[2.,1.],[2.,1.]])
        bundle=dict(trained_through=pd.Timestamp('2023-12-31'),poisson=poisson,model=model,columns=columns,selected='test')
        request=dict(as_of='2024-01-10T12:00:00Z',fixtures=[
            dict(id=1,home='A',away='B',kickoff_time='2024-01-20T12:00:00Z'),
            dict(id=2,home='A',away='B',kickoff_time='2024-01-25T12:00:00Z')])
        with patch('fixture_batch.joblib.load',return_value=bundle) as load, patch('fixture_batch.load_matches',return_value=matches):
            result=forecast_batch(request)
        self.assertEqual(load.call_count,1)
        self.assertEqual(model.predict_proba.call_count,1)
        self.assertEqual(poisson.predict_rates.call_count,1)
        self.assertEqual(result['data_through'],'2024-01-08')
        self.assertAlmostEqual(result['fixtures']['1']['home_clean_sheet'],np.exp(-1))
        features=poisson.predict_rates.call_args.args[0]
        self.assertEqual(features.iloc[0]['home_history_count'],2)
        self.assertEqual(features.iloc[1]['home_history_count'],2)

    def test_historical_model_leakage_rejected(self):
        with patch('fixture_batch.joblib.load',return_value={'trained_through':pd.Timestamp('2026-05-24')}):
            with self.assertRaises(ValueError):
                forecast_batch({'as_of':'2025-01-01T00:00:00Z','fixtures':[]})


if __name__=='__main__': unittest.main()
