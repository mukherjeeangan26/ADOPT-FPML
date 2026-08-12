"""Default NumPy MLP and interfaces for alternative ML trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

Array = np.ndarray


class FittedModel(Protocol):
    input_names: Sequence[str]
    output_names: Sequence[str]

    def predict(self, inputs: pd.DataFrame | Array) -> pd.DataFrame: ...

    @property
    def parameter_count(self) -> int: ...


class MLTrainer(Protocol):
    def fit(self, inputs: pd.DataFrame, targets: pd.DataFrame) -> FittedModel: ...


@dataclass(frozen=True)
class MLTrainingConfig:
    hidden_neurons: int = 8
    epochs: int = 1200
    learning_rate: float = 2.0e-3
    l2: float = 1.0e-5
    batch_size: int | None = None
    patience: int = 150
    validation_fraction: float = 0.15
    seed: int = 7


@dataclass
class NumpyMLP:
    input_names: list[str]
    output_names: list[str]
    W1: Array
    b1: Array
    W2: Array
    b2: Array
    x_mean: Array
    x_std: Array
    y_mean: Array
    y_std: Array
    train_loss: float = float("nan")
    val_loss: float = float("nan")

    def predict(self, inputs: pd.DataFrame | Array) -> pd.DataFrame:
        X = inputs.loc[:, self.input_names].to_numpy(dtype=float) if isinstance(inputs, pd.DataFrame) else np.asarray(inputs, dtype=float)
        Xn = (X - self.x_mean) / self.x_std
        Yn = np.tanh(Xn @ self.W1 + self.b1) @ self.W2 + self.b2
        values = Yn * self.y_std + self.y_mean
        index = inputs.index if isinstance(inputs, pd.DataFrame) else None
        return pd.DataFrame(values, columns=self.output_names, index=index)

    @property
    def parameter_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


@dataclass
class NumpyMLPTrainer:
    """One-hidden-layer tanh network trained with Adam and early stopping."""

    config: MLTrainingConfig = MLTrainingConfig()

    def fit(self, inputs: pd.DataFrame, targets: pd.DataFrame) -> NumpyMLP:
        cfg = self.config
        X, Y = inputs.to_numpy(dtype=float), targets.to_numpy(dtype=float)
        if len(X) < 3:
            raise ValueError("At least three samples are required for training")
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(X))
        n_val = max(1, int(round(cfg.validation_fraction * len(X))))
        n_val = min(n_val, len(X) - 1)
        val_idx, train_idx = order[:n_val], order[n_val:]
        x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0)
        y_mean, y_std = Y[train_idx].mean(0), Y[train_idx].std(0)
        x_std = np.where(x_std < 1e-12, 1.0, x_std)
        y_std = np.where(y_std < 1e-12, 1.0, y_std)
        Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std
        d, h, o = X.shape[1], cfg.hidden_neurons, Y.shape[1]
        W1 = rng.normal(0, np.sqrt(2 / max(d + h, 1)), (d, h))
        b1, W2, b2 = np.zeros(h), rng.normal(0, np.sqrt(2 / max(h + o, 1)), (h, o)), np.zeros(o)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        best = [p.copy() for p in params]
        best_val, stale, step = float("inf"), 0, 0
        batch = cfg.batch_size or len(train_idx)
        for _ in range(cfg.epochs):
            shuffled = rng.permutation(train_idx)
            for start in range(0, len(shuffled), batch):
                ix = shuffled[start : start + batch]
                xb, yb = Xn[ix], Yn[ix]
                hidden = np.tanh(xb @ W1 + b1)
                error = hidden @ W2 + b2 - yb
                scale = 2.0 / max(error.size, 1)
                dW2 = scale * hidden.T @ error + 2 * cfg.l2 * W2
                db2 = scale * error.sum(0)
                dh = (scale * error @ W2.T) * (1 - hidden**2)
                dW1 = xb.T @ dh + 2 * cfg.l2 * W1
                db1 = dh.sum(0)
                step += 1
                for j, (p, g) in enumerate(zip(params, [dW1, db1, dW2, db2])):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    p -= cfg.learning_rate * (m[j] / (1 - 0.9**step)) / (np.sqrt(v[j] / (1 - 0.999**step)) + 1e-8)
            pred_val = np.tanh(Xn[val_idx] @ W1 + b1) @ W2 + b2
            val = float(np.mean((pred_val - Yn[val_idx]) ** 2))
            if val < best_val - 1e-10:
                best_val, stale, best = val, 0, [p.copy() for p in params]
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        W1, b1, W2, b2 = best
        train_pred = np.tanh(Xn[train_idx] @ W1 + b1) @ W2 + b2
        train_loss = float(np.mean((train_pred - Yn[train_idx]) ** 2))
        return NumpyMLP(list(inputs.columns), list(targets.columns), W1, b1, W2, b2, x_mean, x_std, y_mean, y_std, train_loss, best_val)


@dataclass
class SklearnTrainer:
    """Adapter for any scikit-learn-style regressor with ``fit``/``predict``."""

    estimator: object

    def fit(self, inputs: pd.DataFrame, targets: pd.DataFrame) -> "SklearnModel":
        from sklearn.base import clone

        fitted = clone(self.estimator).fit(inputs.to_numpy(), targets.to_numpy())
        return SklearnModel(fitted, list(inputs.columns), list(targets.columns))


@dataclass
class SklearnModel:
    estimator: object
    input_names: list[str]
    output_names: list[str]

    def predict(self, inputs: pd.DataFrame | Array) -> pd.DataFrame:
        X = inputs.loc[:, self.input_names].to_numpy() if isinstance(inputs, pd.DataFrame) else np.asarray(inputs)
        values = np.asarray(self.estimator.predict(X))
        if values.ndim == 1:
            values = values[:, None]
        return pd.DataFrame(values, columns=self.output_names, index=getattr(inputs, "index", None))

    @property
    def parameter_count(self) -> int:
        for attr in ("coef_", "coefs_"):
            if hasattr(self.estimator, attr):
                values = getattr(self.estimator, attr)
                values = values if isinstance(values, list) else [values]
                count = sum(np.asarray(value).size for value in values)
                intercept = getattr(self.estimator, "intercepts_", getattr(self.estimator, "intercept_", []))
                intercept = intercept if isinstance(intercept, list) else [intercept]
                return int(count + sum(np.asarray(value).size for value in intercept))
        raise ValueError("Set a parameter-counting wrapper for this estimator before using AICc")
