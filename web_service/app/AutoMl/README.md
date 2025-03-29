# AutoHPO: Automated Hyperparameter Optimization Framework

A Python framework for comparing hyperparameter optimization (HPO) methods. Currently implements Bayesian Optimization and Evolutionary Strategy approaches. Designed for easy extension with new optimization algorithms and machine learning models.

**Note:** This project is under active development. New features and HPO methods will be added periodically.

## Features

- **Diverse HPO Methods**
  - Bayesian Optimization with Gaussian Processes
  - Evolutionary Strategy with mutation/crossover operations
- **Model Agnostic** - Works with any scikit-learn compatible estimator
- **Multi-task Support** - Handles both classification and regression problems
- **Extensible Architecture** - Easy to add new HPO algorithms
- **Optimization History Tracking** - Compare convergence patterns across methods
