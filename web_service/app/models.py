models = {
        'RandomForest': {
            'n_estimators': {'type': 'int', 'default': 100, 'description': 'Количество деревьев в лесу'},
            'max_depth': {'type': 'int', 'default': None, 'description': 'Максимальная глубина дерева'},
            'batch_size': {'type': 'int', 'default': 16, 'description': 'Размер батча'}
        },
        'SVM': {
            'C': {'type': 'float', 'default': 1.0, 'description': 'Регуляризационный параметр'},
            'gamma': {'type': 'float', 'default': 'scale', 'description': 'Параметр ядра'},
            'batch_size': {'type': 'int', 'default': 16, 'description': 'Размер батча'}
        }
}