import os
import time
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Fitter:
    def __init__(self, model_name, params, dataset):
        self.model_name = model_name
        self.params = params
        self.dataset = dataset

        # Универсальная директория для хранения моделей (без упоминания пользователей)
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')

    def fit(self):
        file_name = self.dataset.file_name
        file_path = self.dataset.file_path
        df = pd.read_csv(file_path)

        # Предполагается, что последний столбец — это целевая переменная
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Универсальный выбор модели через словарь конструкторов
        model_constructors = {
            'RandomForest': lambda params: RandomForestClassifier(**params),
            'SVM': lambda params: svm.SVC(**params),
            'LogisticRegression': lambda params: LogisticRegression(**params),
            'KNN': lambda params: KNeighborsClassifier(**params)
        }

        if self.model_name not in model_constructors:
            raise ValueError(f"Model {self.model_name} is not supported.")

        model = model_constructors[self.model_name](self.params)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        accuracy = model.score(X_test, y_test)

        with mlflow.start_run() as run:
            mlflow.log_params(self.params)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_param("model_name", self.model_name)
            # Логируем информацию о датасете: имя и id
            mlflow.log_param("dataset", f"{self.dataset.file_name} (id:{self.dataset.id})")

# Расширенная конфигурация моделей с заполненными описаниями, значениями по умолчанию и вариантами выбора для категориальных признаков
models = {
    'RandomForest': {
        'n_estimators': {
            'type': 'int',
            'default': 100,
            'description': 'Количество деревьев в лесу'
        },
        'max_depth': {
            'type': 'int',
            'default': 20,
            'description': 'Максимальная глубина дерева'
        },
    },
    'SVM': {
        'C': {
            'type': 'float',
            'default': 1.0,
            'description': 'Параметр регуляризации (стоимость ошибки)'
        },
        'kernel': {
            'type': 'str',
            'default': 'rbf',
            'description': 'Ядро SVM (выберите из списка)',
            'options': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    },
    'LogisticRegression': {
        'C': {
            'type': 'float',
            'default': 1.0,
            'description': 'Обратная величина регуляризации'
        },
        'max_iter': {
            'type': 'int',
            'default': 100,
            'description': 'Максимальное число итераций для сходимости'
        }
    },
    'KNN': {
        'n_neighbors': {
            'type': 'int',
            'default': 5,
            'description': 'Количество соседей для классификации'
        },
        'weights': {
            'type': 'str',
            'default': 'uniform',
            'description': 'Метод взвешивания (выберите из списка)',
            'options': ['uniform', 'distance']
        }
    }
}
