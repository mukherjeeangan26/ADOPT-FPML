# Replacing ML, selection, and pruning components

## ML model class and training solver

The default `NumpyMLPTrainer` uses one tanh hidden layer, Adam, a validation split, and early stopping. Configure it with `MLTrainingConfig`.

Any replacement trainer needs `fit(inputs, targets)`. Its returned fitted model needs `predict(inputs)`, `input_names`, `output_names`, and `parameter_count`. AICc relies on the last field; define it carefully for regularized, shared, or nonparametric models.

`SklearnTrainer` adapts a scikit-learn regressor:

```python
from sklearn.neural_network import MLPRegressor
from adopt_fpml.ml import SklearnTrainer

trainer = SklearnTrainer(MLPRegressor(hidden_layer_sizes=(16,), random_state=7))
optimizer = ADOPTFPML(fp_model, ml_trainer=trainer)
```

For PyTorch, JAX, TensorFlow, Gaussian processes, symbolic regression, or custom solvers, implement the same small protocol. Baseline algorithms are deliberately not copied into this repository.

## Input selection

The paper default is `PearsonSelector`. Built-in alternatives are:

- `SpearmanSelector` for monotonic nonlinear relationships;
- `MutualInformationSelector` for more general dependence (requires the `sklearn` extra).

A custom selector only needs `rank(candidates, targets) -> list[str]`. Sensitivity-based and variance-based selectors are appropriate extensions when a simulator or perturbation design is available.

## Branch pruning

```python
from adopt_fpml import PruningConfig, SearchConfig

greedy = PruningConfig(strategy="greedy")                 # paper default
patient = PruningConfig(strategy="patience", patience=2)
windowed = PruningConfig(strategy="moving_window", window=3)

config = SearchConfig(pruning=patient)
```

Patience and moving-window policies can avoid terminating on a single poor candidate, at the cost of additional training and a larger multiple-comparison search. Report the policy and parameters whenever results are published.

## Objective and output selection

Inject a callable `(prediction, truth, parameter_count) -> float` to replace AICc. A replacement should still penalize complexity and be comparable across candidates.

The current optimizer targets every unresolved output at a stage and accepts the subset meeting thresholds. Output clustering, subset enumeration, or multiobjective output selection can be implemented above the optimizer by running documented target groups. Treat these as extensions, not reproductions of the paper default.

