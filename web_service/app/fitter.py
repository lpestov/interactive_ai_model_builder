import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

class Fitter:
    def __init__(self, model_name, params, dataset):
        self.model_name = model_name
        self.params = params
        self.dataset = dataset

        if not os.path.exists('user_models'):
            os.makedirs('user_models')

    def fit(self):
        file_name = self.dataset.split()[-2][:-1]
        file_path = self.dataset.split()[-1][:-1]
        df = pd.read_csv(file_path)

        # Предполагается, что последний столбец - это целевая переменная
        X = df.iloc[:, :-1]  # Все столбцы, кроме последнего
        y = df.iloc[:, -1]  # Последний столбец как целевая переменная
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if self.model_name == 'RandomForest':
            model = RandomForestClassifier(**self.params)  # Распаковка параметров
        elif self.model_name == 'SVM':
            model = svm.SVC()


        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        accuracy = model.score(X_test, y_test)

        with mlflow.start_run() as run:
            mlflow.log_params(self.params)
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("user_id", "123")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_param("model_name", self.model_name)


models = {
        'RandomForest': {
            'n_estimators': {'type': 'int', 'default': 100, 'description': 'Количество деревьев в лесу'},
            'max_depth': {'type': 'int', 'default': None, 'description': 'Максимальная глубина дерева'},
        },
        'SVM': {
            'C': {'type': 'float', 'default': 1.0, 'description': 'Регуляризационный параметр'},
        }
}







