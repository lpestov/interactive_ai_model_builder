import numpy as np
from warnings import catch_warnings, simplefilter

class EvolutionaryStrategyHPO:
    def __init__(self, f, param_space, population_size=20,
                 mutation_variance=0.1, verbose=1, random_state=None,
                 generations=10, mutation_rate=0.25, mutation_ratio=0.75, elite_ratio=0.2):
        """
        f: Objective function to minimize/maximize. Must accept parameters from param_space as kwargs
        param_space: List of dictionaries defining search space parameters (name, type, low/high/categories)
        population_size: Number of individuals in each generation. Default: 50
        mutation_variance: Scaling factor for mutation step size. Default: 0.1
        verbose: Verbosity level (0 = silent, 1 = basic, 2 = detailed). Default: 0
        random_state: Seed for random number generator. Default: None
        generations: Number of evolutionary iterations. Default: 20
        mutation_rate: Probability of mutating individual parameter. Default: 0.25
        mutation_ratio: Probability of applying mutation to offspring. Default: 0.75
        elite_ratio: Proportion of top performers preserved between generations. Default: 0.2
        """

        self.f = f
        self.param_space = param_space
        self.population_size = population_size
        self.mutation_variance = mutation_variance
        self.verbose = verbose
        self.random_state = random_state
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_ratio = mutation_ratio
        self.elite_ratio = elite_ratio

        self.rng = np.random.RandomState(random_state)
        self.history = []
        self.fitness_cache = {}

        # Валидация параметров
        for param in self.param_space:
            if param['type'] in ['integer', 'float']:
                if param['low'] >= param['high']:
                    raise ValueError(f"Invalid range for {param['name']}")
            elif param['type'] == 'categorical' and not param.get('categories'):
                raise ValueError(f"Invalid categories for {param['name']}")

    def _initialize_individual(self):
        individual = {}
        for param in self.param_space:
            name = param['name']
            if param['type'] == 'float':
                individual[name] = self.rng.uniform(param['low'], param['high'])
            elif param['type'] == 'integer':
                individual[name] = self.rng.randint(param['low'], param['high'] + 1)
            elif param['type'] == 'categorical':
                individual[name] = self.rng.choice(param['categories'])
        return individual

    def _mutate(self, individual):
        mutated = individual.copy()
        for param in self.param_space:
            name = param['name']
            if self.rng.rand() < self.mutation_rate:
                if param['type'] == 'float':
                    delta = self.mutation_variance * (param['high'] - param['low'])
                    new_val = mutated[name] + self.rng.uniform(-delta, delta)
                    mutated[name] = np.clip(new_val, param['low'], param['high'])
                elif param['type'] == 'integer':
                    delta = max(1, int(self.mutation_variance * (param['high'] - param['low'])))
                    new_val = mutated[name] + self.rng.randint(-delta, delta + 1)
                    mutated[name] = int(np.clip(new_val, param['low'], param['high']))
                elif param['type'] == 'categorical':
                    mutated[name] = self.rng.choice(param['categories'])
        return mutated

    def _crossover(self, parent1, parent2):
        child = parent1.copy()
        crossover_params = self.rng.choice(
            self.param_space,
            size=self.rng.randint(1, len(self.param_space) + 1),
            replace=False
        )
        for param in crossover_params:
            child[param['name']] = parent2[param['name']]
        return child

    def _evaluate(self, individual):
        params = {}
        for param in self.param_space:
            name = param['name']
            val = individual[name]
            params[name] = val

        key = tuple(sorted(params.items()))
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        with catch_warnings():
            simplefilter("ignore")
            score = self.f(**params)

        self.fitness_cache[key] = score
        if self.verbose > 1 :
            print(f"Probe: {params} → Score: {score:.4f}")
        return score

    def optimize(self):
        population = [self._initialize_individual() for _ in range(self.population_size)]
        elite_size = max(2, int(self.population_size * self.elite_ratio))

        for gen in range(self.generations):
            fitness = [self._evaluate(ind) for ind in population]
            elite_indices = np.argsort(fitness)[-elite_size:]
            elites = [population[i] for i in elite_indices]

            offspring = []
            for _ in range(self.population_size - elite_size):
                parents = self.rng.choice(elites, 2, replace=False)
                child = self._crossover(parents[0], parents[1])
                if self.rng.rand() < self.mutation_ratio:
                    child = self._mutate(child)
                offspring.append(child)

            population = elites + offspring
            best_fitness = np.max(fitness)
            self.history.append(best_fitness)
            if self.verbose:
                print(f"Generation {gen + 1}/{self.generations} | Best: {best_fitness:.4f}")


        best_idx = np.argmax([self._evaluate(ind) for ind in population])
        best_params = population[best_idx]

        if self.verbose > 0:
            print("═" * 50)
            print("Best params for Evolution Strategy:")
            for k, v in best_params.items():
                print(f"▸ {k:20} : {v}")
            print(f"Best result: {max(self.history):.4f}")
            print("═" * 50)

        return best_params, self.history