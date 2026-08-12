"""E1: dynamic three-CSTR acetone cracking example."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np
import pandas as pd

from ..serialization import load_numpy_mlp

Array = np.ndarray
INPUTS = ["CA0", "T_in", "v", "q1_dot", "q2_dot", "q3_dot"]
OUTPUTS = ["CA", "CB", "CC", "T_out"]
MEASURED_COLUMNS = [
    "CA_final_measured",
    "CB_final_measured",
    "CC_final_measured",
    "T_out_final_measured",
]


@dataclass(frozen=True)
class E1PlantConstants:
    A_true: float = 1.0e10
    Ea_over_R_true: float = 34222.0
    delta_H: float = 60000.0
    Cp_A: float = 163.0
    n_stages: int = 3
    V_total: float = 1.0e-2
    reaction_order: float = 1.2
    dt: float = 1.0
    substeps_per_sample: int = 8
    ca_eps: float = 1.0e-8
    t_eps: float = 300.0

    @property
    def volumes(self) -> Array:
        return np.full(self.n_stages, self.V_total / self.n_stages)


@dataclass(frozen=True)
class E1FPConstants:
    K: float = 1.0e10 * math.exp(-34222.0 / 1350.0)
    delta_H: float = 60000.0
    Cp_A: float = 163.0
    n_stages: int = 3
    V_total: float = 1.0e-2
    reaction_order: float = 1.0
    dt: float = 1.0
    substeps_per_sample: int = 8
    ca_eps: float = 1.0e-8
    t_eps: float = 300.0

    @property
    def volumes(self) -> Array:
        return np.full(self.n_stages, self.V_total / self.n_stages)


def _step_signal(n: int, low: float, high: float, hold: int, seed: int, digits: int) -> Array:
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / hold))
    return np.repeat(np.round(rng.uniform(low, high, blocks), digits), hold)[:n]


def _rhs(y: Array, u: dict[str, float], constants: object, k_value: float) -> Array:
    dydt = np.zeros_like(y)
    for stage in range(constants.n_stages):
        ca, cb, cc, temperature = y[stage]
        if stage == 0:
            ca_in, cb_in, cc_in, t_in = u["CA0"], 0.0, 0.0, u["T_in"]
        else:
            ca_in, cb_in, cc_in, t_in = y[stage - 1]
        volume = constants.volumes[stage]
        ca_safe = max(float(ca), constants.ca_eps)
        rate = k_value * ca_safe**constants.reaction_order
        flow = u["v"] / volume
        dydt[stage, 0] = flow * (ca_in - ca) - rate
        dydt[stage, 1] = flow * (cb_in - cb) + rate
        dydt[stage, 2] = flow * (cc_in - cc) + rate
        dydt[stage, 3] = (
            flow * (t_in - temperature)
            + u[f"q{stage + 1}_dot"] / (constants.Cp_A * ca_safe)
            - constants.delta_H * rate / (constants.Cp_A * ca_safe)
        )
    return dydt


def _advance(y: Array, u: dict[str, float], constants: object, k_value: float) -> Array:
    result = y.copy()
    h = constants.dt / constants.substeps_per_sample
    for _ in range(constants.substeps_per_sample):
        k1 = _rhs(result, u, constants, k_value)
        k2 = _rhs(result + 0.5 * h * k1, u, constants, k_value)
        k3 = _rhs(result + 0.5 * h * k2, u, constants, k_value)
        k4 = _rhs(result + h * k3, u, constants, k_value)
        result += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        result[:, :3] = np.clip(result[:, :3], 0, None)
        result[:, 3] = np.clip(result[:, 3], constants.t_eps, None)
    return result


def generate_e1_data(
    n_steps: int = 500,
    hold_time: int = 35,
    seed_base: int = 123,
    q_scale: float = 13.0,
    noise_seed: int = 2026,
    constants: E1PlantConstants = E1PlantConstants(),
) -> pd.DataFrame:
    """Generate the biased/noisy E1 dataset exactly as described in Appendix A."""

    signals = {
        "CA0": _step_signal(n_steps, 40, 100, hold_time, seed_base + 1, 3),
        "T_in": _step_signal(n_steps, 1000, 1400, hold_time, seed_base + 2, 3),
        "v": _step_signal(n_steps, 0.0008, 0.0015, hold_time, seed_base + 3, 6),
        "q1_dot": q_scale * _step_signal(n_steps, 10000, 20000, hold_time, seed_base + 4, 3),
        "q2_dot": q_scale * _step_signal(n_steps, 10000, 20000, hold_time, seed_base + 5, 3),
        "q3_dot": q_scale * _step_signal(n_steps, 10000, 20000, hold_time, seed_base + 6, 3),
    }
    out = pd.DataFrame({"time_index": np.arange(n_steps), **signals})
    clean = np.zeros((n_steps, constants.n_stages, 4))
    state = np.tile(np.array([signals["CA0"][0], 0.1, 0.1, signals["T_in"][0]]), (3, 1))
    clean[0] = state
    for row in range(1, n_steps):
        u = {name: float(signals[name][row]) for name in INPUTS}
        # The plant rate constant is evaluated locally in temperature, so use a
        # dedicated RHS below rather than the fixed-k FP simplification.
        def plant_rhs(values: Array) -> Array:
            dydt = np.zeros_like(values)
            for s in range(constants.n_stages):
                ca, cb, cc, temp = values[s]
                ca_in, cb_in, cc_in, tin = (
                    (u["CA0"], 0.0, 0.0, u["T_in"]) if s == 0 else tuple(values[s - 1])
                )
                ca_safe = max(float(ca), constants.ca_eps)
                rate = constants.A_true * math.exp(-constants.Ea_over_R_true / max(float(temp), constants.t_eps)) * ca_safe**constants.reaction_order
                flow = u["v"] / constants.volumes[s]
                dydt[s] = [
                    flow * (ca_in - ca) - rate,
                    flow * (cb_in - cb) + rate,
                    flow * (cc_in - cc) + rate,
                    flow * (tin - temp) + u[f"q{s + 1}_dot"] / (constants.Cp_A * ca_safe) - constants.delta_H * rate / (constants.Cp_A * ca_safe),
                ]
            return dydt
        h = constants.dt / constants.substeps_per_sample
        for _ in range(constants.substeps_per_sample):
            a = plant_rhs(state)
            b = plant_rhs(state + 0.5 * h * a)
            c = plant_rhs(state + 0.5 * h * b)
            d = plant_rhs(state + h * c)
            state += h * (a + 2 * b + 2 * c + d) / 6
            state[:, :3] = np.clip(state[:, :3], 0, None)
            state[:, 3] = np.clip(state[:, 3], constants.t_eps, None)
        clean[row] = state
    Y = clean[:, -1]
    rng = np.random.default_rng(noise_seed)
    ca0n = (signals["CA0"] - np.mean(signals["CA0"])) / np.std(signals["CA0"])
    tinn = (signals["T_in"] - np.mean(signals["T_in"])) / np.std(signals["T_in"])
    scale = np.maximum(np.std(Y, axis=0), 1e-8)
    driver1 = 0.70 * ca0n - 0.55 * tinn + 0.45 * ca0n * tinn + 0.35 * ca0n**2 - 0.18 * tinn**2
    measured1 = Y + np.column_stack([c * scale[j] * driver1 for j, c in enumerate([0.35, 0.15, 0.35, 0.12])])
    measured1 += rng.normal(0, np.array([0.010, 0.012, 0.007, 0.005]) * scale, Y.shape)
    measured1[:, :3] = np.clip(measured1[:, :3], 0, None)
    measured1[:, 3] = np.clip(measured1[:, 3], constants.t_eps, None)
    normalized = [(measured1[:, j] - measured1[:, j].mean()) / measured1[:, j].std() for j in (0, 1, 3)]
    driver2 = 0.55 * normalized[0] - 0.40 * normalized[1] + 0.35 * normalized[2] + 0.20 * normalized[0] ** 2 - 0.15 * normalized[1] * normalized[2]
    measured = measured1 + np.column_stack([c * scale[j] * driver2 for j, c in enumerate([0.30, 0.40, 0.35, 0.06])])
    measured += rng.normal(0, np.array([0.035, 0.035, 0.035, 0.005]) * scale, Y.shape)
    measured[:, :3] = np.clip(measured[:, :3], 0, None)
    measured[:, 3] = np.clip(measured[:, 3], constants.t_eps, None)
    for s in range(constants.n_stages):
        for j, label in enumerate(("CA", "CB", "CC", "T")):
            out[f"{label}{s + 1}_clean"] = clean[:, s, j]
    for name, values in zip(MEASURED_COLUMNS, measured.T):
        out[name] = values
    return out


@dataclass
class E1FirstPrinciplesModel:
    constants: E1FPConstants = E1FPConstants()
    output_names: tuple[str, ...] = tuple(OUTPUTS)

    def predict(self, inputs: pd.DataFrame, parameters: pd.DataFrame | None = None) -> pd.DataFrame:
        missing = [name for name in INPUTS if name not in inputs]
        if missing:
            raise ValueError(f"E1 FP inputs missing columns: {missing}")
        state = np.tile(np.array([inputs.CA0.iloc[0], 0.1, 0.1, inputs.T_in.iloc[0]]), (3, 1))
        values = np.zeros((len(inputs), 4))
        values[0] = state[-1]
        for row in range(1, len(inputs)):
            u = {name: float(inputs[name].iloc[row]) for name in INPUTS}
            k_value = self.constants.K if parameters is None else max(float(parameters["K"].iloc[row]), 1e-12)
            state = _advance(state, u, self.constants, k_value)
            values[row] = state[-1]
        return pd.DataFrame(values, columns=OUTPUTS, index=inputs.index)


@dataclass
class E1ParameterTargetGenerator:
    """Bounded one-step estimator for the E1 lumped kinetic parameter ``K``."""

    fp_model: E1FirstPrinciplesModel = field(default_factory=E1FirstPrinciplesModel)
    local_search_steps: int = 35
    parameter_names: tuple[str, ...] = ("K",)

    def generate(
        self,
        inputs: pd.DataFrame,
        measured_outputs: pd.DataFrame,
        fp_predictions: pd.DataFrame,
        target_outputs: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        from scipy.optimize import minimize

        constants = self.fp_model.constants
        targets = np.zeros((len(inputs), 1))
        targets[0, 0] = constants.K
        output_indexes = [OUTPUTS.index(name) for name in target_outputs]
        scale = np.maximum(measured_outputs.loc[:, target_outputs].std().to_numpy(), 1e-12)
        log_base = math.log10(max(constants.K, 1e-12))
        bounds = [(log_base - 4, log_base + 4)]
        for row in range(1, len(inputs)):
            previous = measured_outputs.iloc[row - 1]
            state = np.tile(previous.loc[OUTPUTS].to_numpy(dtype=float), (constants.n_stages, 1))
            u = {name: float(inputs[name].iloc[row]) for name in INPUTS}
            truth = measured_outputs.loc[:, target_outputs].iloc[row].to_numpy(dtype=float)

            def objective(value: Array) -> float:
                prediction = _advance(state, u, constants, 10 ** float(value[0]))[-1, output_indexes]
                return float(np.mean(((prediction - truth) / scale) ** 2))

            result = minimize(
                objective,
                x0=np.array([log_base]),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.local_search_steps, "ftol": 1e-10},
            )
            targets[row, 0] = 10 ** (float(result.x[0]) if result.success else log_base)
        return pd.DataFrame(targets, columns=self.parameter_names, index=inputs.index)


def _bundled_model(name: str):
    root = files("adopt_fpml").joinpath("data", "e1", "models")
    return load_numpy_mlp(root.joinpath(f"{name}.npz"), root.joinpath(f"{name}.json"))


def run_e1_pretrained(data: pd.DataFrame) -> pd.DataFrame:
    """Run the retained E1 two-block configuration using portable artifacts."""

    fp = E1FirstPrinciplesModel().predict(data.loc[:, INPUTS])
    block1 = _bundled_model("block_01_series_1")
    features1 = pd.DataFrame(index=data.index)
    for name in block1.input_names:
        features1[name] = fp[name[3:]] if name.startswith("fp_") else data[name]
    pred1 = block1.predict(features1)
    for name in ("CA", "CB", "CC"):
        if name in pred1:
            pred1[name] = pred1[name].clip(lower=0.0)
    block2 = _bundled_model("block_02_parallel")
    features2 = pd.DataFrame(index=data.index)
    for name in block2.input_names:
        if name.startswith("fp_"):
            features2[name] = fp[name[3:]]
        elif name.startswith("series_1_pred_"):
            features2[name] = pred1[name.removeprefix("series_1_pred_")]
        else:
            features2[name] = data[name]
    correction = block2.predict(features2)
    stage2_raw = fp[["CA", "T_out"]] + correction[["CA", "T_out"]].to_numpy()
    stage2_raw["CA"] = stage2_raw["CA"].clip(lower=0.0)
    final = fp.copy()
    final[["CB", "CC"]] = pred1[["CB", "CC"]]
    final[["CA", "T_out"]] = stage2_raw
    return final
