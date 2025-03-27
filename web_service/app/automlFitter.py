# automlFitter.py
import sys
import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '../AutoML'))
from AutoMl.automl.models import BayesianOptimizationHPO, EvolutionaryStrategyHPO


class AutoMLFitter:
    def __init__(self, hpo_method, hpo_params, model_name, param_space_list, dataset, task_type, split_ratio,
                 target_column, scoring):
        self.hpo_method = hpo_method
        self.hpo_params = hpo_params
        self.model_name = model_name
        self.model_param_space = param_space_list
        self.task_type = task_type
        self.split_ratio = split_ratio
        self.target_column = target_column
        self.scoring = scoring

        df = pd.read_csv(dataset.file_path)
        self.dataset = dataset

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=float(split_ratio) / 100)

        # Маппинг для моделей sklearn
        self.model_mapping = {
            # Classification models
            "RandomForestClassifier": RandomForestClassifier,
            "SVC": SVC,
            "LogisticRegression": LogisticRegression,
            "GradientBoostingClassifier": GradientBoostingClassifier,

            # Regression models
            "RandomForestRegressor": RandomForestRegressor,
            "SVR": SVR,
            "GradientBoostingRegressor": GradientBoostingRegressor
        }

        # Маппинг для методов оптимизации гиперпараметров AutoML
        self.hpo_mapping = {
            "Bayesian": BayesianOptimizationHPO,
            "Evolutionary": EvolutionaryStrategyHPO
        }

    def _f(self, **param_space):
        model = self.model_class(**param_space)
        model.fit(self.X_train, self.y_train)
        return self._calculate_score(model)

    def _calculate_score(self, model):
        y_pred = model.predict(self.X_test)
        if self.task_type == 'classification':
            if self.scoring == 'accuracy':
                score = accuracy_score(self.y_test, y_pred)
            elif self.scoring == 'f1':
                score = f1_score(self.y_test, y_pred, average='weighted')
            else:
                score = model.score(self.X_test, self.y_test)
        else:
            if self.scoring == 'r2':
                score = r2_score(self.y_test, y_pred)
            elif self.scoring == 'mse':
                score = mean_squared_error(self.y_test, y_pred)
            else:
                score = model.score(self.X_test, self.y_test)
        return score

    def fit(self):
        start_time = time.time()
        with mlflow.start_run() as run:
            mlflow.set_tag("status", "training")
            mlflow.log_param("scoring", self.scoring)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("dataset", f"{self.dataset.file_name} (id:{self.dataset.id})")
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("train/test ratio", self.split_ratio)
            mlflow.log_param("target_column", self.target_column)
            mlflow.log_param("hpo_method", self.hpo_method)
            mlflow.log_param("hpo_params", self.hpo_params)
            mlflow.set_tag("model_type", "auto_ml")

            self.model_class = self.model_mapping.get(self.model_name)
            self.hpo_class = self.hpo_mapping.get(self.hpo_method)
            hpo_model = self.hpo_class(self._f, self.model_param_space, **self.hpo_params)

            best_params, history = hpo_model.optimize()
            training_time = time.time() - start_time

            model = self.model_class(**best_params)
            model.fit(self.X_train, self.y_train)
            best_score = self._calculate_score(model)

            mlflow.log_metric("score", best_score)
            mlflow.log_metric("training_time", training_time)

            mlflow.log_param("best_params", best_params)
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("status", "finished")