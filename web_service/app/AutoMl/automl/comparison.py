import time
import numpy as np
from sklearn.model_selection import cross_val_score
from warnings import catch_warnings, simplefilter

def compare_hpo_methods(config, X, y):
    task_type = config['task_type']
    model_config = config['model']
    scoring = config['scoring']
    hpo_methods_config = config['hpo_methods']

    # Определение целевой функции
    def objective_function(**params):
        model_class = model_config['class']
        model_params = model_config['fixed_params'].copy()

        for param in model_config['param_space']:
            name = param['name']
            value = params[name]
            if param['type'] == 'integer':
                model_params[name] = int(round(value))
            elif param['type'] == 'float':
                model_params[name] = value
            elif param['type'] == 'categorical':
                model_params[name] = value

        model = model_class(**model_params)
        with catch_warnings():
            simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        return np.mean(scores)

    # Сравнение методов HPO
    results = {}
    for method_name, method_config in hpo_methods_config.items():
        print(f"=== {method_name} ({task_type}) ===")
        HPOClass = method_config['hpo_class']
        hpo_params = method_config.get('hpo_params', {})

        # Инициализация оптимизатора
        hpo = HPOClass(
            f=objective_function,
            param_space=model_config['param_space'],
            **hpo_params
        )

        # Запуск оптимизации
        start_time = time.time()
        best_params, history = hpo.optimize()

        best_score = max(history)
        time_taken = time.time() - start_time

        results[method_name] = {
            'best_score': best_score,
            'time': time_taken,
            'history': history,
            'best_params': best_params
        }

        # Вывод результатов метода
        print(f"Best Score: {best_score:.4f}, Time: {time_taken:.2f}s")
        print("=" * 30 + '\n')

    # Сводка результатов
    print(f"\n=== Final Results ===")
    for method, res in results.items():
        print(f"\n**{method}**")
        print(f"  Best Score: {res['best_score']:.4f}")
        print(f"  Time: {res['time']:.2f}s")
        print("  Best Parameters:")
        for param, value in res['best_params'].items():
            print(f"    {param}: {value}")
    print("\n" + "=" * 50)





