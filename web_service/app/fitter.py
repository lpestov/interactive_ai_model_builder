import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .models import TrainHistory
from .extentions import db

class Fitter:
    def __init__(self, model_name, params, dataset):
        self.model_name = model_name
        self.params = params
        self.dataset = dataset

    def fit(self):
        if self.model_name == 'RandomForest':
            file_name = self.dataset.split()[-2][:-1]
            file_path = self.dataset.split()[-1][:-1]
            self.df = pd.read_csv(file_path)

            # Предполагается, что последний столбец - это целевая переменная
            X = self.df.iloc[:, :-1]  # Все столбцы, кроме последнего
            y = self.df.iloc[:, -1]   # Последний столбец как целевая переменная

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            self.model = RandomForestClassifier(**self.params)  # Распаковка параметров
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)

            history = TrainHistory(self.model_name,file_name, self.params, accuracy)
            db.session.add(history)
            db.session.commit()


models = {
        'RandomForest': {
            'n_estimators': {'type': 'int', 'default': 100, 'description': 'Количество деревьев в лесу'},
            'max_depth': {'type': 'int', 'default': None, 'description': 'Максимальная глубина дерева'},
        },
        'SVM': {
            'C': {'type': 'float', 'default': 1.0, 'description': 'Регуляризационный параметр'},
            'gamma': {'type': 'float', 'default': 'scale', 'description': 'Параметр ядра'},
        }
}