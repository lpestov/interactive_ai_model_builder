import sys
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../AutoML'))
from AutoMl.automl.models import BayesianOptimizationHPO, EvolutionaryStrategyHPO

class AutoMLFitter:
    def __init__(self, hpo_method, hpo_params, model_name, model_param_space, dataset):
        self.hpo_method = hpo_method
        self.hpo_params = hpo_params
        self.model_name = model_name
        self.model_param_space = model_param_space
        self.dataset = pd.read_csv(dataset.file_path)


        # Маппинг для моделей sklearn
        self.model_mapping = {
            "RandomForestClassifier": RandomForestClassifier,
            "SVC": SVC,
            "LogisticRegression": LogisticRegression
        }

        # Маппинг для методов оптимизации гиперпараметров AutoML
        self.hpo_mapping = {
            "Bayesian": BayesianOptimizationHPO,
            "Evolutionary": EvolutionaryStrategyHPO
        }



    def _f(self, **param_space):
        print(param_space)




        model = self.model_class(**param_space)
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy')
        return np.mean(scores)


    def fit(self):
        self.X = self.dataset.iloc[:, :-1]
        self.y = self.dataset.iloc[:, -1]
        self.model_class = self.model_mapping.get(self.model_name)


        self.hpo_class = self.hpo_mapping.get(self.hpo_method)
        model = self.hpo_class(self._f, self.model_param_space, **self.hpo_params)

        print(type(self.model_param_space))
        print(self.hpo_params)

        best_params, history = model.optimize()

        print(best_params, history)



        '''
        with mlflow.start_run() as run:
            mlflow.log_params()
            mlflow.log_metric()
            mlflow.sklearn.log_model(model, "model")'''