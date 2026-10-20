"""Serializable outcome models with a common predict_proba interface."""
import numpy as np
from scipy.stats import skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def poisson_probabilities(home_rate, away_rate):
    """Sum independent Poisson scorelines via their exact Skellam difference law.

    No arbitrary score cutoff or discarded tail probability is needed.
    """
    home_rate = np.maximum(np.asarray(home_rate, dtype=float), 1e-8)
    away_rate = np.maximum(np.asarray(away_rate, dtype=float), 1e-8)
    probabilities = np.column_stack([
        skellam.sf(0, home_rate, away_rate),
        skellam.pmf(0, home_rate, away_rate),
        skellam.cdf(-1, home_rate, away_rate)])
    probabilities = np.maximum(probabilities, 0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class PoissonOutcome:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, x, goals):
        self.models = []
        for col in ['FTHG', 'FTAG']:
            model = make_pipeline(SimpleImputer(add_indicator=True, keep_empty_features=True),
                StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000))
            model.fit(x, goals[col])
            self.models.append(model)
        return self

    def predict_rates(self, x):
        return np.column_stack([m.predict(x) for m in self.models])

    def predict_proba(self, x):
        rates = self.predict_rates(x)
        return poisson_probabilities(rates[:, 0], rates[:, 1])


class ProbabilityBlend:
    def __init__(self, classifier, poisson, poisson_weight):
        self.classifier = classifier
        self.poisson = poisson
        self.poisson_weight = poisson_weight

    def predict_proba(self, x):
        probabilities = ((1-self.poisson_weight)*np.asarray(self.classifier.predict_proba(x), dtype=float)
                         + self.poisson_weight*self.poisson.predict_proba(x))
        return probabilities / probabilities.sum(axis=1, keepdims=True)
