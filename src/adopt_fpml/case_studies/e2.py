"""E2: cooler plus isothermal flash example and reconstructed generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np
import pandas as pd

from ..serialization import load_numpy_mlp

Array = np.ndarray
INPUTS = ["CA_feed", "CB_feed", "CC_feed", "T_feed", "v"]
OUTPUTS = ["F_vap", "F_liq", "T_vap", "T_liq", "y_A", "y_B", "y_C", "x_A", "x_B", "x_C"]
MEASURED_COLUMNS = [f"{name}_measured" for name in OUTPUTS]


@dataclass
class E2Constants:
    R: float = 8.314
    T_ref: float = 330.0
    P_cooler: float = 2.0
    P_flash: float = 2.0
    T_coolant: float = 285.0
    P_ref: float = 1.0
    beta_cooler_target: float = 0.35
    K_ref: Array = field(default_factory=lambda: np.array([0.15, 12.0, 0.35]))
    dH_K: Array = field(default_factory=lambda: np.array([29000.0, 8000.0, 22000.0]))
    Cp_v: Array = field(default_factory=lambda: np.array([75.0, 36.0, 44.0]))
    Cp_l: Array = field(default_factory=lambda: np.array([125.0, 55.0, 90.0]))
    dH_vap: Array = field(default_factory=lambda: np.array([29000.0, 8000.0, 22000.0]))


@dataclass(frozen=True)
class E2FPParameters:
    UA: float = 14.147425712387479
    K_A: float = 0.075
    K_B: float = 9.214225970119335
    K_C: float = 0.648368955345655


def _normalize(values: Array) -> Array:
    values = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return values / values.sum()


def _rr(beta: float, z: Array, K: Array) -> float:
    return float(np.sum(z * (K - 1) / (1 + beta * (K - 1))))


def _bisect(function, lower: float, upper: float, iterations: int = 60) -> float:
    f_low, f_high = function(lower), function(upper)
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0:
        return 0.5 * (lower + upper)
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        f_mid = function(middle)
        if abs(f_mid) < 1e-10 or abs(upper - lower) < 1e-8:
            return middle
        if f_low * f_mid <= 0:
            upper, f_high = middle, f_mid
        else:
            lower, f_low = middle, f_mid
    return 0.5 * (lower + upper)


def _phase_split(z: Array, K: Array) -> tuple[float, Array, Array, str]:
    f0, f1 = _rr(0, z, K), _rr(1, z, K)
    if f0 <= 0:
        return 0.0, z.copy(), _normalize(K * z), "liquid"
    if f1 >= 0:
        return 1.0, _normalize(z / K), z.copy(), "vapor"
    beta = _bisect(lambda value: _rr(value, z, K), 0, 1)
    x = _normalize(z / (1 + beta * (K - 1)))
    return beta, x, _normalize(K * x), "two-phase"


def _cooler_k(temperature: float, constants: E2Constants) -> Array:
    exponent = -(constants.dH_K / constants.R) * (1 / temperature - 1 / constants.T_ref)
    return np.clip(constants.K_ref * constants.P_ref / constants.P_cooler * np.exp(np.clip(exponent, -50, 50)), 1e-10, 1e10)


def _cooler_temperature(z: Array, constants: E2Constants) -> float:
    function = lambda temperature: _rr(constants.beta_cooler_target, z, _cooler_k(temperature, constants))
    grid = np.linspace(180, 650, 500)
    values = np.array([function(value) for value in grid])
    for i in range(len(grid) - 1):
        if values[i] == 0:
            return float(grid[i])
        if np.isfinite(values[i : i + 2]).all() and values[i] * values[i + 1] < 0:
            return _bisect(function, float(grid[i]), float(grid[i + 1]))
    return float(grid[np.nanargmin(np.abs(values))])


def _clean_row(
    ca: float, cb: float, cc: float, temperature: float, velocity: float, constants: E2Constants
) -> dict[str, object]:
    flows = np.array([ca, cb, cc]) * velocity
    total = max(float(flows.sum()), 1e-12)
    z = _normalize(flows)
    tc = _cooler_temperature(z, constants)
    beta_c, xc, yc, _ = _phase_split(z, _cooler_k(tc, constants))
    cp_rate = total * float(z @ constants.Cp_v)
    ua = -cp_rate * math.log((tc - constants.T_coolant) / (temperature - constants.T_coolant))
    feed_h = float(np.sum(flows * (constants.Cp_v * (temperature - constants.T_ref) + constants.dH_vap)))
    vapor_h = float(yc @ (constants.Cp_v * (tc - constants.T_ref) + constants.dH_vap))
    liquid_h = float(xc @ (constants.Cp_l * (tc - constants.T_ref)))
    cooler_h = total * (beta_c * vapor_h + (1 - beta_c) * liquid_h)
    flash_k = constants.K_ref * constants.P_ref / constants.P_flash
    beta, x, y, phase = _phase_split(z, flash_k)
    return {
        "flows": flows,
        "total": total,
        "z": z,
        "tc": tc,
        "beta_c": beta_c,
        "ua": ua,
        "duty": feed_h - cooler_h,
        "xc": xc,
        "yc": yc,
        "beta": beta,
        "phase": phase,
        "x": x,
        "y": y,
        "fv": beta * total,
        "fl": (1 - beta) * total,
    }


def generate_e2_data(e1_feed: pd.DataFrame, noise_seed: int = 42) -> pd.DataFrame:
    """Generate E2 measurements from an E1 outlet trajectory.

    This restores the generator missing from the supplied E2 archive using
    Appendix B, the retained authoritative data, seed 42, and its verified draw
    order. ``e1_feed`` may use E1 measured names or canonical E2 feed names.
    """

    aliases = {
        "CA_feed": "CA_final_measured",
        "CB_feed": "CB_final_measured",
        "CC_feed": "CC_final_measured",
        "T_feed": "T_out_final_measured",
        "v": "v",
    }
    feed = pd.DataFrame(index=e1_feed.index)
    for canonical, e1_name in aliases.items():
        source = canonical if canonical in e1_feed else e1_name
        if source not in e1_feed:
            raise ValueError(f"E2 generator missing feed column {canonical!r}/{e1_name!r}")
        feed[canonical] = e1_feed[source].to_numpy()
    constants = E2Constants()
    clean = [
        _clean_row(row.CA_feed, row.CB_feed, row.CC_feed, row.T_feed, row.v, constants)
        for row in feed.itertuples(index=False)
    ]
    n = len(feed)
    rng = np.random.default_rng(noise_seed)
    fv_true = np.array([row["fv"] for row in clean])
    fl_true = np.array([row["fl"] for row in clean])
    fv_measured = np.clip(fv_true + rng.normal(0, 0.003 * fv_true, n), 0, None)
    fl_measured = np.clip(fl_true + rng.normal(0, 0.003 * fl_true, n), 0, None)
    x_true = np.vstack([row["x"] for row in clean])
    y_true = np.vstack([row["y"] for row in clean])
    x_measured = np.vstack([_normalize(row) for row in x_true + rng.normal(0, 0.001, (n, 3))])
    y_measured = np.vstack([_normalize(row) for row in y_true + rng.normal(0, 0.001, (n, 3))])
    tc = np.array([row["tc"] for row in clean])
    tv_measured = tc + rng.normal(0, 0.1, n)
    tl_measured = tc + rng.normal(0, 0.1, n)
    out = pd.DataFrame({
        "step_index": np.arange(n),
        "original_time_index": e1_feed["time_index"].to_numpy() if "time_index" in e1_feed else e1_feed.index.to_numpy(),
        "v": feed.v,
        "CA_feed_measured": feed.CA_feed,
        "CB_feed_measured": feed.CB_feed,
        "CC_feed_measured": feed.CC_feed,
        "T_feed_measured": feed.T_feed,
    })
    flows = np.vstack([row["flows"] for row in clean])
    z = np.vstack([row["z"] for row in clean])
    out[["F_A_feed", "F_B_feed", "F_C_feed"]] = flows
    out["F_feed"] = [row["total"] for row in clean]
    out[["z_A", "z_B", "z_C"]] = z
    out["P_cooler"] = constants.P_cooler
    out["T_cooler_out"] = tc
    out["beta_cooler"] = [row["beta_c"] for row in clean]
    out["UA_required"] = [row["ua"] for row in clean]
    out["Q_cooler"] = [row["duty"] for row in clean]
    out[["x_A_cooler", "x_B_cooler", "x_C_cooler"]] = np.vstack([row["xc"] for row in clean])
    out[["y_A_cooler", "y_B_cooler", "y_C_cooler"]] = np.vstack([row["yc"] for row in clean])
    out["P_flash"] = constants.P_flash
    out["T_flash_true"] = tc
    out["beta_flash"] = [row["beta"] for row in clean]
    out["phase_state"] = [row["phase"] for row in clean]
    out[["K_A_flash", "K_B_flash", "K_C_flash"]] = constants.K_ref * constants.P_ref / constants.P_flash
    out["F_vap_true"], out["F_vap_measured"] = fv_true, fv_measured
    out["F_liq_true"], out["F_liq_measured"] = fl_true, fl_measured
    for i, label in enumerate("ABC"):
        out[f"y_{label}_true"] = y_true[:, i]
    for i, label in enumerate("ABC"):
        out[f"y_{label}_measured"] = y_measured[:, i]
    for i, label in enumerate("ABC"):
        out[f"x_{label}_true"] = x_true[:, i]
    for i, label in enumerate("ABC"):
        out[f"x_{label}_measured"] = x_measured[:, i]
    out["T_vap_true"], out["T_vap_measured"] = tc, tv_measured
    out["T_liq_true"], out["T_liq_measured"] = tc, tl_measured
    return out


@dataclass
class E2FirstPrinciplesModel:
    parameters: E2FPParameters = E2FPParameters()
    constants: E2Constants = field(default_factory=E2Constants)
    output_names: tuple[str, ...] = tuple(OUTPUTS)

    def predict(self, inputs: pd.DataFrame, parameters: pd.DataFrame | None = None) -> pd.DataFrame:
        missing = [name for name in INPUTS if name not in inputs]
        if missing:
            raise ValueError(f"E2 FP inputs missing columns: {missing}")
        result = []
        for i, row in enumerate(inputs.loc[:, INPUTS].itertuples(index=False)):
            p = self.parameters if parameters is None else E2FPParameters(**{name: max(float(parameters[name].iloc[i]), 1e-8) for name in ("UA", "K_A", "K_B", "K_C")})
            flows = np.array([row.CA_feed, row.CB_feed, row.CC_feed]) * row.v
            total, z = max(float(flows.sum()), 1e-12), _normalize(flows)
            cp_rate = total * float(z @ self.constants.Cp_v)
            tc = self.constants.T_coolant + (row.T_feed - self.constants.T_coolant) * math.exp(-p.UA / max(cp_rate, 1e-12))
            K = np.array([p.K_A, p.K_B, p.K_C])
            beta, x, y, _ = _phase_split(z, K)
            result.append([beta * total, (1 - beta) * total, tc, tc, *y, *x])
        return pd.DataFrame(result, columns=OUTPUTS, index=inputs.index)


@dataclass
class E2ParameterTargetGenerator:
    """Bounded row-wise estimator for E2 ``UA`` and three equilibrium ratios."""

    fp_model: E2FirstPrinciplesModel = field(default_factory=E2FirstPrinciplesModel)
    local_search_steps: int = 18
    parameter_names: tuple[str, ...] = ("UA", "K_A", "K_B", "K_C")

    def generate(
        self,
        inputs: pd.DataFrame,
        measured_outputs: pd.DataFrame,
        fp_predictions: pd.DataFrame,
        target_outputs: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        from scipy.optimize import minimize

        base = np.array(
            [
                self.fp_model.parameters.UA,
                self.fp_model.parameters.K_A,
                self.fp_model.parameters.K_B,
                self.fp_model.parameters.K_C,
            ]
        )
        log_base = np.log10(np.clip(base, 1e-8, None))
        bounds = [(value - 2, value + 2) for value in log_base]
        current = log_base.copy()
        scale = np.maximum(measured_outputs.loc[:, target_outputs].std().to_numpy(), 1e-12)
        targets = np.zeros((len(inputs), 4))
        for row in range(len(inputs)):
            row_inputs = inputs.iloc[[row]]
            truth = measured_outputs.loc[:, target_outputs].iloc[row].to_numpy(dtype=float)

            def objective(values: Array) -> float:
                theta = 10 ** np.asarray(values, dtype=float)
                model = E2FirstPrinciplesModel(E2FPParameters(*theta), self.fp_model.constants)
                prediction = model.predict(row_inputs).loc[:, target_outputs].iloc[0].to_numpy()
                return float(np.mean(((prediction - truth) / scale) ** 2))

            result = minimize(
                objective,
                x0=current,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.local_search_steps, "ftol": 1e-10},
            )
            if result.success and np.isfinite(result.fun):
                current = np.asarray(result.x, dtype=float)
            targets[row] = 10**current
        return pd.DataFrame(targets, columns=self.parameter_names, index=inputs.index)


def _bundled_model(name: str):
    root = files("adopt_fpml").joinpath("data", "e2", "models")
    return load_numpy_mlp(root.joinpath(f"{name}.npz"), root.joinpath(f"{name}.json"))


def _canonical_inputs(data: pd.DataFrame) -> pd.DataFrame:
    aliases = {"CA_feed": "CA_feed_measured", "CB_feed": "CB_feed_measured", "CC_feed": "CC_feed_measured", "T_feed": "T_feed_measured", "v": "v"}
    return pd.DataFrame({name: data[name if name in data else alias] for name, alias in aliases.items()}, index=data.index)


def run_e2_pretrained(data: pd.DataFrame) -> pd.DataFrame:
    """Run the retained E2 two-block configuration using portable artifacts."""

    X = _canonical_inputs(data)
    fp_model = E2FirstPrinciplesModel()
    fp = fp_model.predict(X)
    block1 = _bundled_model("block_01_series_1")
    features1 = pd.DataFrame({name: fp[name[3:]] for name in block1.input_names}, index=data.index)
    pred1 = block1.predict(features1)
    block2 = _bundled_model("block_02_integrated")
    features2 = pd.DataFrame(index=data.index)
    for name in block2.input_names:
        features2[name] = fp[name[3:]] if name.startswith("fp_") else X[name]
    theta = block2.predict(features2)
    integrated = fp_model.predict(X, theta)
    final = fp.copy()
    stage1_outputs = [name for name in block1.output_names if name in final]
    final.loc[:, stage1_outputs] = pred1.loc[:, stage1_outputs]
    final[["F_liq", "y_A"]] = integrated[["F_liq", "y_A"]]
    return final
