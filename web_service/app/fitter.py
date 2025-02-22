import os

import joblib
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .extentions import db
from .models import TrainHistory


class Fitter:
    def __init__(self, model_name, params, dataset):
        self.model_name = model_name
        self.params = params
        self.dataset = dataset

        if not os.path.exists("user_models"):
            os.makedirs("user_models")

    def fit(self):
        file_name = self.dataset.split()[-2][:-1]
        file_path = self.dataset.split()[-1][:-1]
        df = pd.read_csv(file_path)

        # Предполагается, что последний столбец - это целевая переменная
        X = df.iloc[:, :-1]  # Все столбцы, кроме последнего
        y = df.iloc[:, -1]  # Последний столбец как целевая переменная
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if self.model_name == "RandomForest":
            model = RandomForestClassifier(**self.params)  # Распаковка параметров
        elif self.model_name == "SVM":
            model = svm.SVC()

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Сохранение модели в указанную директорию
        model_file_path = os.path.join("user_models", f"{file_name}.joblib")
        joblib.dump(model, model_file_path)

        history = TrainHistory(
            self.model_name,
            file_name,
            self.params,
            test_accuracy=accuracy,
            model_path=model_file_path,
        )
        db.session.add(history)
        db.session.commit()


models = {
    "RandomForest": {
        "n_estimators": {
            "type": "int",
            "default": 100,
            "description": "Количество деревьев в лесу",
        },
        "max_depth": {
            "type": "int",
            "default": None,
            "description": "Максимальная глубина дерева",
        },
    },
    "SVM": {
        "C": {
            "type": "float",
            "default": 1.0,
            "description": "Регуляризационный параметр",
        },
    },
}
