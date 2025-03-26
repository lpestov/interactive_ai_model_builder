from ..automl.models import BayesianOptimizationHPO, EvolutionaryStrategyHPO
from ..automl.comparison import compare_hpo_methods


from sklearn.ensemble import GradientBoostingRegressor
config_regression = {
    'task_type': 'regression',
    'model': {
        'class': GradientBoostingRegressor,
        'fixed_params': {
            'random_state': 53,
            'loss': 'squared_error'
        },
        'param_space': [
            {'name': 'n_estimators', 'type': 'integer', 'low': 50, 'high': 300},
            {'name': 'learning_rate', 'type': 'float', 'low': 0.01, 'high': 0.3},
            {'name': 'max_depth', 'type': 'integer', 'low': 3, 'high': 8},
            {'name': 'min_samples_split', 'type': 'integer', 'low': 2, 'high': 15},
            {'name': 'min_samples_leaf', 'type': 'integer', 'low': 1, 'high': 5},
            {'name': 'subsample', 'type': 'float', 'low': 0.6, 'high': 1.0},
        ]
    },
    'scoring': 'neg_mean_squared_error',
    'hpo_methods': {
        'Bayesian': {
            'hpo_class': BayesianOptimizationHPO,
            'hpo_params': {
                'verbose': 1,
                'random_state': 152,
            }
        },
        'Evolutionary': {
            'hpo_class': EvolutionaryStrategyHPO,
            'hpo_params': {
                'verbose': 1,
                'random_state': 152,
            }
        }
    }
}

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, noise=0.1, n_features=8, random_state=50)
compare_hpo_methods(config_regression,X,y)