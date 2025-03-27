import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

class Fitter:
    def __init__(self, task_type, model_name, params, dataset, split_ratio, target_column, scoring):
        self.task_type = task_type
        self.model_name = model_name
        self.params = params
        self.dataset = dataset
        self.scoring = scoring
        self.target_column = target_column
        self.split_ratio = split_ratio

    def fit(self):
        with mlflow.start_run() as run:
            mlflow.set_tag("status", "training")
            mlflow.log_params(self.params)
            mlflow.log_param("scoring", self.scoring)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("dataset", f"{self.dataset.file_name} (id:{self.dataset.id})")
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("train/test ratio", self.split_ratio)
            mlflow.log_param("target_column", self.target_column)


            file_path = self.dataset.file_path
            df = pd.read_csv(file_path)
            X = df.drop(self.target_column, axis = 1)
            y = df[self.target_column]
            if self.task_type == 'regression':
                y = pd.to_numeric(y, errors='raise')

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.split_ratio / 100)
            if self.task_type == 'classification':
                model_constructors = {
                    'RandomForestClassifier': lambda p: RandomForestClassifier(**p),
                    'SVM': lambda p: svm.SVC(**p),
                    'LogisticRegression': lambda p: LogisticRegression(**p),
                    'KNN': lambda p: KNeighborsClassifier(**p)
                }
            else:
                model_constructors = {
                    'RandomForestRegressor': lambda p: RandomForestRegressor(**p),
                    'SVR': lambda p: svm.SVR(**p),
                    'LinearRegression': lambda p: LinearRegression(**p),
                    'KNNRegressor': lambda p: KNeighborsRegressor(**p)
                }
            if self.model_name not in model_constructors:
                raise ValueError(f"Model {self.model_name} is not supported.")
            model = model_constructors[self.model_name](self.params)

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            if self.task_type == 'classification':
                if self.scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif self.scoring == 'f1':
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    score = model.score(X_test, y_test)
            else:
                if self.scoring == 'r2':
                    score = r2_score(y_test, y_pred)
                elif self.scoring == 'mse':
                    score = mean_squared_error(y_test, y_pred)
                else:
                    score = model.score(X_test, y_test)


            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("score", score)
            mlflow.log_metric("training_time", training_time)
            mlflow.set_tag("status", "finished")
            mlflow.set_tag("model_type", "classic_ml")

classification_models = {
    'RandomForestClassifier': {
        'n_estimators': {
            'type': 'int',
            'default': 100,
            'description': 'Number of trees in the forest'
        },
        'max_depth': {
            'type': 'int',
            'default': 20,
            'description': 'Maximum tree depth'
        },
        'min_samples_split' : {
            'type' : 'int',
            'default' : 2,
            'description' : 'Minimum number of samples required to split an internal node'
        }
    },
    'SVM': {
        'C': {
            'type': 'float',
            'default': 1.0,
            'description': 'Regularization parameter'
        },
        'kernel': {
            'type': 'str',
            'default': 'rbf',
            'description': 'SVM kernel',
            'options': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    },
    'LogisticRegression': {
        'C': {
            'type': 'float',
            'default': 1.0,
            'description': 'Inverse regularization strength'
        },
        'max_iter': {
            'type': 'int',
            'default': 100,
            'description': 'Maximum number of iterations'
        }
    },
    'KNN': {
        'n_neighbors': {
            'type': 'int',
            'default': 5,
            'description': 'Number of neighbors'
        },
        'weights': {
            'type': 'str',
            'default': 'uniform',
            'description': 'Weighting method',
            'options': ['uniform', 'distance']
        }
    }
}

regression_models = {
    'RandomForestRegressor': {
        'n_estimators': {
            'type': 'int',
            'default': 100,
            'description': 'Number of trees in the forest'
        },
        'max_depth': {
            'type': 'int',
            'default': 20,
            'description': 'Maximum tree depth'
        }
    },
    'SVR': {
        'C': {
            'type': 'float',
            'default': 1.0,
            'description': 'Regularization parameter'
        },
        'kernel': {
            'type': 'str',
            'default': 'rbf',
            'description': 'SVR kernel',
            'options': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    },
    'LinearRegression': {},
    'KNNRegressor': {
        'n_neighbors': {
            'type': 'int',
            'default': 5,
            'description': 'Number of neighbors'
        },
        'weights': {
            'type': 'str',
            'default': 'uniform',
            'description': 'Weighting method',
            'options': ['uniform', 'distance']
        }
    }
}
