# Sklearn Backend Examples

This directory contains examples demonstrating Hyperactive's sklearn backend integration. This backend provides direct access to scikit-learn's mature and optimized hyperparameter search implementations (GridSearchCV and RandomizedSearchCV) through Hyperactive's unified interface.

## Quick Start

Run any example directly:
```bash
python grid_search_example.py
python random_search_example.py
```

## Available Algorithms

| Algorithm | Sklearn Equivalent | Best For | Characteristics |
|-----------|-------------------|----------|-----------------|
| [GridSearchSk](grid_search_example.py) | GridSearchCV | Small discrete spaces | Exhaustive, deterministic |
| [RandomSearchSk](random_search_example.py) | RandomizedSearchCV | General purpose | Efficient sampling |

## When to Use Sklearn Backend

The sklearn backend is ideal when you:
- Want to leverage sklearn's mature, well-tested implementations
- Need compatibility with existing sklearn pipelines and workflows
- Require sklearn's specific cross-validation and scoring features
- Prefer sklearn's parameter specification format
- Want to benefit from sklearn's optimized parallel execution

## Sklearn vs Other Backends

### Advantages of Sklearn Backend:
- **Mature implementation**: Battle-tested in production environments
- **Sklearn integration**: Natural fit for sklearn-based workflows
- **Efficient execution**: Optimized for sklearn estimators
- **Familiar interface**: Uses sklearn's parameter specification style
- **Built-in features**: Cross-validation, scoring, and parallelization

### When to Consider Other Backends:
- **Advanced algorithms**: GFO backend offers more sophisticated optimization
- **Optuna features**: Optuna backend provides state-of-the-art Bayesian optimization
- **Custom objectives**: Non-sklearn experiments may benefit from other backends

## Parameter Space Specification

### Grid Search - Discrete Values Only
```python
param_space = {
    "n_estimators": [10, 50, 100, 200],     # Explicit discrete values
    "max_depth": [5, 10, 15, None],         # Include special values
    "criterion": ["gini", "entropy"],       # Categorical choices
    "bootstrap": [True, False],             # Boolean options
}
```

### Random Search - Ranges and Distributions
```python
param_space = {
    "n_estimators": (10, 200),              # Continuous range (sampled as int)
    "max_depth": (1, 20),                   # Integer range
    "min_samples_split": (2, 20),           # Integer range
    "max_features": ["sqrt", "log2", None], # Discrete choices
}
```

## Integration with Sklearn Pipelines

The sklearn backend works seamlessly with sklearn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter space with pipeline syntax
param_space = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'scaler__with_mean': [True, False],
}

# Use with Hyperactive
experiment = SklearnCvExperiment(estimator=pipeline, X=X, y=y, cv=5)
optimizer = GridSearchSk(param_space=param_space, experiment=experiment)
```

## Cross-Validation Strategies

Leverage sklearn's flexible cross-validation:

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Stratified K-Fold (maintains class distribution)
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=cv_strat)

# Time Series Split (for temporal data)
cv_time = TimeSeriesSplit(n_splits=5)
experiment = SklearnCvExperiment(estimator=estimator, X=X, y=y, cv=cv_time)
```

## Scoring Options

Use sklearn's extensive scoring options:

```python
# Single metric
experiment = SklearnCvExperiment(
    estimator=estimator,
    X=X, y=y,
    cv=3,
    scoring='f1_macro'  # Use F1 score with macro averaging
)

# Multiple metrics (requires custom handling)
experiment = SklearnCvExperiment(
    estimator=estimator,
    X=X, y=y,
    cv=3,
    scoring='accuracy'  # Primary scoring metric
)
```

## Performance Optimization

### Parallel Execution
Both algorithms support parallel execution through sklearn's `n_jobs` parameter:

```python
# Note: n_jobs is handled internally by sklearn backend
# The degree of parallelization depends on sklearn's implementation
```

### Memory Efficiency
For large datasets, consider:
- Reducing cross-validation folds for faster iteration
- Using smaller parameter grids for GridSearchSk
- Limiting n_trials for RandomSearchSk

## Comparison: Grid vs Random Search

### GridSearchSk
**Pros:**
- Exhaustive coverage of parameter space
- Deterministic and reproducible results
- Guarantees finding optimal solution within the grid
- Good for small, well-defined parameter spaces

**Cons:**
- Exponential growth in computation time
- Requires discrete parameter values
- Not suitable for high-dimensional spaces

### RandomSearchSk
**Pros:**
- Scales well to high-dimensional spaces
- Can sample from continuous distributions
- Often finds good solutions with fewer evaluations
- More flexible parameter specification

**Cons:**
- Stochastic - results may vary between runs
- No guarantee of finding optimal solution
- Requires choosing appropriate number of trials

## Best Practices

1. **Start with RandomSearchSk**: Good default choice for most problems
2. **Use GridSearchSk for small spaces**: When you can afford exhaustive search
3. **Leverage sklearn features**: Use appropriate CV strategies and scoring
4. **Set random_state**: For reproducible results
5. **Monitor convergence**: Check if more trials would help
6. **Consider other backends**: For more advanced optimization needs

## Integration Examples

### With Preprocessing Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PCA

pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier())
])

param_space = {
    'pca__n_components': [5, 10, 15, 20],
    'classifier__n_estimators': (50, 300),
    'classifier__max_depth': [5, 10, None]
}
```

### With Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', RandomForestClassifier())
])

param_space = {
    'feature_selection__k': [5, 10, 15, 'all'],
    'classifier__n_estimators': [50, 100, 200]
}
```

## Further Reading

- [Sklearn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Sklearn RandomizedSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Sklearn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html)
- [Cross-validation Strategies](https://scikit-learn.org/stable/modules/cross_validation.html)
