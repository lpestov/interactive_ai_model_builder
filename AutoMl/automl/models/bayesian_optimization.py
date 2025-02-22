import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from warnings import catch_warnings, simplefilter
import warnings

class BayesianOptimizationHPO:
    def __init__(self, f, param_space, verbose=1, random_state=None,
                 init_points=10, n_iter=50, acq='ucb', kappa=2.576, xi=0.0,
                 n_candidates=500):
        """
        f: Objective function to minimize/maximize. Must accept parameters from param_space as kwargs
        param_space: List of dictionaries defining search space parameters (name, type, low/high/categories)
        verbose: Verbosity level (0 = silent, 1 = basic, 2 = detailed). Default: 1
        random_state: Seed for random number generator. Default: None
        init_points: Number of initial random evaluations before Bayesian optimization begins. Default: 10
        n_iter: Number of Bayesian optimization iterations. Default: 50
        acq: Acquisition function type ('ucb' or 'ei'). Default: 'ucb'
        kappa: Exploration-exploitation parameter for UCB. Higher values promote exploration. Default: 2.576
        xi: Exploration parameter for EI, controls improvement threshold. Default: 0.0
        n_candidates: Number of candidate samples to evaluate for acquisition function optimization. Default: 500
        """
        self.f = f
        self.param_space = param_space
        self.verbose = verbose
        self.random_state = random_state
        self.init_points = init_points
        self.n_iter = n_iter
        self.acq = acq
        self.kappa = kappa
        self.xi = xi
        self.n_candidates = n_candidates

        self._space = []
        self._values = []
        self.rng = np.random.RandomState(random_state)
        self.pbounds = {}
        self.categorical_mapping = {}

        for param in param_space:
            name = param['name']
            param_type = param['type']
            if param_type == 'categorical':
                if 'categories' not in param:
                    raise ValueError(f"Parameter {name} of type 'categorical' must have 'categories' list.")
                categories = param['categories']
                if not isinstance(categories, list) or len(categories) == 0:
                    raise ValueError(f"Parameter {name} 'categories' must be a non-empty list.")
                self.categorical_mapping[name] = categories.copy()
                low = 0
                high = len(categories) - 1
                self.pbounds[name] = (low, high)
            else:
                if 'low' not in param or 'high' not in param:
                    raise ValueError(f"Parameter {name} missing 'low' or 'high'")
                low = param['low']
                high = param['high']
                if low >= high:
                    raise ValueError(f"Invalid range for {name} (low >= high)")
                if param_type not in ['integer', 'float']:
                    raise ValueError(f"Unsupported parameter type {param_type} for {name}")
                self.pbounds[name] = (low, high)

        # Issue warning if categorical parameters are present
        if self.categorical_mapping:
            warnings.warn(
                "Warning: The use of categorical parameters is strongly discouraged in Bayesian Optimization. "
                "Categorical variables are converted to integers, which may not be suitable for the underlying Gaussian Process model. "
                "Consider using continuous or integer variables instead.", UserWarning
            )

        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=random_state
        )
        self._prime_subscriptions()

    def optimize(self):
        self.init(self.init_points)

        for i in range(self.n_iter):
            with catch_warnings():
                simplefilter("ignore")
                self.probe(self._next())

            if self.verbose >= 1:
                print(f"Iteration {i + 1}/{self.n_iter} | Best: {self.max['target']:.4f}")

        best_params = {}
        for param in self.param_space:
            name = param['name']
            value = self.max['params'][name]
            if param['type'] == 'integer':
                best_params[name] = int(round(value))
            elif param['type'] == 'categorical':
                categories = param['categories']
                index = int(np.round(value))
                index = np.clip(index, 0, len(categories) - 1)
                best_params[name] = categories[index]
            else:
                best_params[name] = value

        history = self._values.copy()

        if self.verbose > 0:
            print("═" * 50)
            print("Best params for Bayesian Optimization:")
            for k, v in best_params.items():
                print(f"▸ {k:20} : {v}")
            print(f"\nBest result: {self.max['target']:.4f}")
            print("═" * 50)

        return best_params, history

    def _next(self):
        self.gp.fit(self._space, self._values)
        best_acq = -np.inf
        best_x = None

        for _ in range(self.n_candidates):
            candidate = np.array([self.rng.uniform(low, high) for (low, high) in self._bounds])

            mu, sigma = self.gp.predict([candidate], return_std=True)

            if self.acq == 'ucb':
                acq_value = mu + self.kappa * sigma
            elif self.acq == 'ei':
                improvement = mu - self.max['target'] - self.xi
                z = improvement / sigma if sigma > 1e-12 else 0.0
                acq_value = (improvement * norm.cdf(z) + sigma * norm.pdf(z)) if sigma > 1e-12 else 0.0

            if acq_value > best_acq:
                best_acq = acq_value
                best_x = candidate

        return best_x

    def _prime_subscriptions(self):
        self._bounds = np.array([v for k, v in self.pbounds.items()])
        self.dim = len(self.pbounds)
        self.initialized = False

    def init(self, init_points):
        for _ in range(init_points):
            x = np.array([self.rng.uniform(low, high) for (low, high) in self._bounds])
            self.probe(x)

    def probe(self, x):
        params = dict(zip(self.pbounds.keys(), x))
        for name in self.categorical_mapping:
            continuous_val = params[name]
            categories = self.categorical_mapping[name]
            index = int(np.round(continuous_val))
            index = np.clip(index, 0, len(categories) - 1)
            params[name] = categories[index]

        target = self.f(**params)

        self._space.append(x)
        self._values.append(target)

        if self.verbose > 1:
            print(f"Probe: {params} → Score: {target:.4f}")

    @property
    def max(self):
        idx = np.argmax(self._values)
        params_dict = dict(zip(self.pbounds.keys(), self._space[idx]))
        return {
            'params': params_dict,
            'target': self._values[idx]
        }