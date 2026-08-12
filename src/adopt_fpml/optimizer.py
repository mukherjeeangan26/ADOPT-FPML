"""Stage-wise ADOPT-FPML branch-and-prune optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .fp import FirstPrinciplesModel
from .metrics import (
    AICcObjective,
    MetricThreshold,
    Objective,
    normalize_thresholds,
    relative_rmse_by_output,
    rmse_by_output,
)
from .ml import FittedModel, MLTrainer, NumpyMLPTrainer
from .pruning import PruningConfig
from .selectors import InputSelector, PearsonSelector


class Structure(str, Enum):
    """The four route configurations in the reduced superstructure."""

    PARALLEL = "parallel"
    SERIES_1 = "series_1"
    SERIES_2 = "series_2"
    INTEGRATED = "integrated"


class ParameterTargetGenerator(Protocol):
    """Generate row-wise FP parameter targets for Series II/integrated blocks."""

    parameter_names: Sequence[str]

    def generate(
        self,
        inputs: pd.DataFrame,
        measured_outputs: pd.DataFrame,
        fp_predictions: pd.DataFrame,
        target_outputs: Sequence[str],
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SearchConfig:
    max_stages: int = 4
    max_global_aicc_relative_increase: float = 0.0
    max_inputs_per_branch: int | None = None
    max_candidate_library_size: int | None = None
    propagated_input_cap: int = 4
    relevance_threshold: float = 0.02
    redundancy_threshold: float = 0.95
    structures: tuple[Structure, ...] = (
        Structure.PARALLEL,
        Structure.SERIES_1,
        Structure.SERIES_2,
        Structure.INTEGRATED,
    )
    pruning: PruningConfig = PruningConfig()
    require_all_stage_outputs_resolved: bool = False


@dataclass
class CandidateResult:
    stage: int
    structure: Structure
    input_names: list[str]
    target_outputs: list[str]
    resolved_outputs: list[str]
    predictions: pd.DataFrame
    model: FittedModel
    local_aicc: float
    global_aicc: float
    parameter_count: int
    rmse: dict[str, float]
    relative_rmse: dict[str, float]
    parameter_targets: pd.DataFrame | None = None


@dataclass
class DiscoveryResult:
    fp_predictions: pd.DataFrame
    fp_resolved_outputs: list[str]
    fp_rmse: dict[str, float]
    stages: list[CandidateResult]
    unresolved_outputs: list[str]
    final_predictions: pd.DataFrame
    search_history: pd.DataFrame
    stage_summaries: pd.DataFrame
    termination_reason: str

    @property
    def optimal_configuration(self) -> list[dict[str, object]]:
        return [
            {
                "stage": block.stage,
                "structure": block.structure.value,
                "inputs": list(block.input_names),
                "target_outputs": list(block.target_outputs),
                "resolved_outputs": list(block.resolved_outputs),
                "local_aicc": block.local_aicc,
                "global_aicc": block.global_aicc,
                "parameter_count": block.parameter_count,
            }
            for block in self.stages
        ]

    def summary(self) -> str:
        lines = ["ADOPT-FPML discovery result"]
        lines.append(f"FP-resolved outputs: {self.fp_resolved_outputs or 'none'}")
        for block in self.stages:
            lines.append(
                f"Stage {block.stage}: {block.structure.value}; inputs={block.input_names}; "
                f"resolved={block.resolved_outputs}; local AICc={block.local_aicc:.6g}"
            )
        lines.append(f"Unresolved outputs: {self.unresolved_outputs or 'none'}")
        lines.append(f"Termination: {self.termination_reason}")
        return "\n".join(lines)


@dataclass
class ADOPTFPML:
    """Discover an optimal hybrid information-flow configuration.

    The paper algorithm is the default: Pearson ranking, sequential inclusion,
    local AICc selection, greedy pruning, and a strict global-AICc continuation
    tolerance. All major choices are injected objects or :class:`SearchConfig`
    fields so users can replace them without editing the optimizer.
    """

    fp_model: FirstPrinciplesModel
    ml_trainer: MLTrainer = field(default_factory=NumpyMLPTrainer)
    input_selector: InputSelector = field(default_factory=PearsonSelector)
    parameter_target_generator: ParameterTargetGenerator | None = None
    objective: Objective = field(default_factory=AICcObjective)

    def fit(
        self,
        inputs: pd.DataFrame,
        measured_outputs: pd.DataFrame,
        thresholds: Mapping[str, float | MetricThreshold],
        config: SearchConfig = SearchConfig(),
    ) -> DiscoveryResult:
        X = self._validate_inputs(inputs)
        Y = self._validate_outputs(measured_outputs)
        if not X.index.equals(Y.index):
            raise ValueError("inputs and measured_outputs must have identical row indexes")
        rules = normalize_thresholds(thresholds, list(Y.columns))
        y_fp = self.fp_model.predict(X).loc[:, Y.columns]
        self._validate_finite(y_fp, "FP predictions")
        fp_rmse = rmse_by_output(y_fp, Y)
        fp_resolved = self._resolved(y_fp, Y, rules)
        unresolved = [name for name in Y if name not in fp_resolved]
        final_predictions = y_fp.copy()
        candidate_library = X.copy()
        stages: list[CandidateResult] = []
        history: list[dict[str, object]] = []
        summaries: list[dict[str, object]] = []
        total_k = 0
        global_aicc = self.objective(final_predictions, Y, total_k)
        termination = "all outputs satisfied their thresholds" if not unresolved else ""

        for stage in range(1, config.max_stages + 1):
            if not unresolved:
                break
            unresolved_before = list(unresolved)
            targets = Y.loc[:, unresolved_before]
            library = candidate_library.loc[:, ~candidate_library.columns.duplicated()].copy()
            ranked = self.input_selector.rank(library, targets)
            if config.max_candidate_library_size is not None:
                ranked = ranked[: config.max_candidate_library_size]
            if not ranked:
                termination = "candidate input library was empty"
                break

            fp_features = y_fp.add_prefix("fp_")
            theta_targets = None
            theta_error = None
            if self.parameter_target_generator is not None and any(
                structure in (Structure.SERIES_2, Structure.INTEGRATED)
                for structure in config.structures
            ):
                try:
                    theta_targets = self.parameter_target_generator.generate(
                        X, Y, y_fp, unresolved_before
                    )
                    if not theta_targets.index.equals(X.index):
                        raise ValueError("parameter targets must preserve the input index")
                    self._validate_finite(theta_targets, "parameter targets")
                except Exception as exc:  # retained in the transparent search log
                    theta_error = str(exc)

            best: CandidateResult | None = None
            for structure in config.structures:
                if structure in (Structure.SERIES_2, Structure.INTEGRATED) and theta_targets is None:
                    history.append(
                        self._history_failure(
                            stage,
                            structure,
                            [],
                            "parameter-target generator unavailable"
                            if self.parameter_target_generator is None
                            else f"parameter-target generation failed: {theta_error}",
                        )
                    )
                    continue
                needs_fp_input = structure in (Structure.SERIES_1, Structure.INTEGRATED)
                if needs_fp_input:
                    structure_library = pd.concat([library, fp_features], axis=1)
                    structure_library = structure_library.loc[
                        :, ~structure_library.columns.duplicated()
                    ]
                    ranked_all = self.input_selector.rank(structure_library, targets)
                    ranked_fp = self.input_selector.rank(fp_features, targets)
                    if not ranked_fp:
                        continue
                    required = ranked_fp[0]
                    sequence = [required] + [name for name in ranked_all if name != required]
                else:
                    structure_library, sequence = library, list(ranked)
                if config.max_candidate_library_size is not None:
                    sequence = sequence[: config.max_candidate_library_size]

                selected: list[str] = []
                branch_values: list[float] = []
                max_inputs = config.max_inputs_per_branch or len(sequence)
                for input_name in sequence[:max_inputs]:
                    selected.append(input_name)
                    try:
                        candidate = self._train_candidate(
                            stage,
                            structure,
                            structure_library.loc[:, selected],
                            X,
                            Y,
                            y_fp,
                            unresolved_before,
                            theta_targets,
                            rules,
                            final_predictions,
                            total_k,
                            config.require_all_stage_outputs_resolved,
                        )
                    except Exception as exc:
                        history.append(
                            self._history_failure(stage, structure, selected, str(exc))
                        )
                        continue
                    branch_values.append(candidate.local_aicc)
                    history.append(
                        {
                            "stage": stage,
                            "structure": structure.value,
                            "inputs": tuple(selected),
                            "status": "ok",
                            "local_aicc": candidate.local_aicc,
                            "global_aicc": candidate.global_aicc,
                            "resolved_outputs": tuple(candidate.resolved_outputs),
                            "rmse": candidate.rmse,
                        }
                    )
                    # Manuscript selection rule: within-stage minimum local AICc
                    # among candidates that resolve at least one new output.
                    if candidate.resolved_outputs and (
                        best is None or candidate.local_aicc < best.local_aicc
                    ):
                        best = candidate
                    if config.pruning.should_stop(branch_values):
                        break

            if best is None:
                termination = "no admissible branch resolved a remaining output"
                summaries.append(
                    self._stage_summary(stage, False, None, unresolved_before, termination, global_aicc)
                )
                break

            relative_change = (best.global_aicc - global_aicc) / max(abs(global_aicc), 1e-12)
            if relative_change >= config.max_global_aicc_relative_increase:
                termination = (
                    "best local branch exceeded the allowed relative global AICc increase"
                )
                summaries.append(
                    self._stage_summary(stage, False, best, unresolved_before, termination, global_aicc)
                )
                break

            stages.append(best)
            total_k += best.parameter_count
            global_aicc = best.global_aicc
            for name in unresolved_before:
                final_predictions[name] = best.predictions[name]
            unresolved = [name for name in unresolved_before if name not in best.resolved_outputs]
            propagated = self._screen_propagated(
                best.predictions.add_prefix(f"stage{stage}_"),
                candidate_library,
                Y.loc[:, unresolved] if unresolved else Y,
                config,
            )
            candidate_library = pd.concat([candidate_library, propagated], axis=1)
            summaries.append(
                self._stage_summary(stage, True, best, unresolved, "retained", global_aicc, propagated)
            )
            if not unresolved:
                termination = "all outputs satisfied their thresholds"

        if unresolved and not termination:
            termination = f"maximum stage count ({config.max_stages}) reached"
        return DiscoveryResult(
            fp_predictions=y_fp,
            fp_resolved_outputs=fp_resolved,
            fp_rmse=fp_rmse,
            stages=stages,
            unresolved_outputs=unresolved,
            final_predictions=final_predictions,
            search_history=pd.DataFrame(history),
            stage_summaries=pd.DataFrame(summaries),
            termination_reason=termination,
        )

    def _train_candidate(
        self,
        stage: int,
        structure: Structure,
        branch_inputs: pd.DataFrame,
        raw_inputs: pd.DataFrame,
        truth: pd.DataFrame,
        fp_predictions: pd.DataFrame,
        targets: list[str],
        theta_targets: pd.DataFrame | None,
        rules: Mapping[str, MetricThreshold],
        current_predictions: pd.DataFrame,
        existing_k: int,
        require_all: bool,
    ) -> CandidateResult:
        if structure == Structure.PARALLEL:
            train_targets = truth.loc[:, targets] - fp_predictions.loc[:, targets]
            model = self.ml_trainer.fit(branch_inputs, train_targets)
            predictions = fp_predictions.loc[:, targets] + model.predict(branch_inputs).to_numpy()
        elif structure == Structure.SERIES_1:
            model = self.ml_trainer.fit(branch_inputs, truth.loc[:, targets])
            predictions = model.predict(branch_inputs)
            predictions.columns = targets
        else:
            assert theta_targets is not None
            model = self.ml_trainer.fit(branch_inputs, theta_targets)
            parameters = model.predict(branch_inputs)
            parameters.columns = list(theta_targets.columns)
            predictions = self.fp_model.predict(raw_inputs, parameters).loc[:, targets]
        predictions = pd.DataFrame(predictions, columns=targets, index=truth.index)
        self._validate_finite(predictions, "candidate predictions")
        local_aicc = self.objective(predictions, truth.loc[:, targets], model.parameter_count)
        resolved = self._resolved(predictions, truth.loc[:, targets], rules)
        if require_all and len(resolved) != len(targets):
            resolved = []
        trial = current_predictions.copy()
        for name in targets:
            trial[name] = predictions[name]
        global_aicc = self.objective(trial, truth, existing_k + model.parameter_count)
        return CandidateResult(
            stage=stage,
            structure=structure,
            input_names=list(branch_inputs.columns),
            target_outputs=list(targets),
            resolved_outputs=resolved,
            predictions=predictions,
            model=model,
            local_aicc=local_aicc,
            global_aicc=global_aicc,
            parameter_count=model.parameter_count,
            rmse=rmse_by_output(predictions, truth.loc[:, targets]),
            relative_rmse=relative_rmse_by_output(predictions, truth.loc[:, targets]),
            parameter_targets=theta_targets.copy() if theta_targets is not None else None,
        )

    @staticmethod
    def _validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("inputs must be a non-empty pandas DataFrame")
        result = frame.copy()
        nonnumeric = [name for name in result if not pd.api.types.is_numeric_dtype(result[name])]
        if nonnumeric:
            raise ValueError(f"All input columns must be numeric: {nonnumeric}")
        ADOPTFPML._validate_finite(result, "inputs")
        return result

    def _validate_outputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("measured_outputs must be a non-empty pandas DataFrame")
        expected = list(self.fp_model.output_names)
        missing = [name for name in expected if name not in frame]
        if missing:
            raise ValueError(f"Measured outputs missing FP output columns: {missing}")
        result = frame.loc[:, expected].copy()
        self._validate_finite(result, "measured outputs")
        return result

    @staticmethod
    def _validate_finite(frame: pd.DataFrame, label: str) -> None:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contain NaN or infinite values")

    @staticmethod
    def _resolved(
        prediction: pd.DataFrame,
        truth: pd.DataFrame,
        rules: Mapping[str, MetricThreshold],
    ) -> list[str]:
        return [
            name
            for name in prediction
            if rules[name].evaluate(prediction[name].to_numpy(), truth[name].to_numpy())[0]
        ]

    @staticmethod
    def _history_failure(
        stage: int, structure: Structure, inputs: Sequence[str], reason: str
    ) -> dict[str, object]:
        return {
            "stage": stage,
            "structure": structure.value,
            "inputs": tuple(inputs),
            "status": f"skipped/failed: {reason}",
            "local_aicc": np.nan,
            "global_aicc": np.nan,
            "resolved_outputs": tuple(),
        }

    @staticmethod
    def _screen_propagated(
        features: pd.DataFrame,
        existing: pd.DataFrame,
        targets: pd.DataFrame,
        config: SearchConfig,
    ) -> pd.DataFrame:
        scored: list[tuple[float, str]] = []
        for name in features:
            x = features[name].to_numpy(dtype=float)
            relevance_values = []
            for target in targets:
                y = targets[target].to_numpy(dtype=float)
                relevance_values.append(
                    0.0 if np.std(x) < 1e-12 or np.std(y) < 1e-12 else abs(np.corrcoef(x, y)[0, 1])
                )
            relevance = float(np.mean(relevance_values))
            redundancy = 0.0
            for prior in existing:
                z = existing[prior].to_numpy(dtype=float)
                if np.std(x) >= 1e-12 and np.std(z) >= 1e-12:
                    redundancy = max(redundancy, float(abs(np.corrcoef(x, z)[0, 1])))
            if relevance >= config.relevance_threshold and redundancy <= config.redundancy_threshold:
                scored.append((relevance, name))
        keep = [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))]
        return features.loc[:, keep[: config.propagated_input_cap]].copy()

    @staticmethod
    def _stage_summary(
        stage: int,
        retained: bool,
        candidate: CandidateResult | None,
        unresolved: Sequence[str],
        reason: str,
        reference_global_aicc: float,
        propagated: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        return {
            "stage": stage,
            "retained": retained,
            "termination_reason": reason,
            "structure": candidate.structure.value if candidate else "",
            "inputs": tuple(candidate.input_names) if candidate else tuple(),
            "resolved_outputs": tuple(candidate.resolved_outputs) if candidate else tuple(),
            "unresolved_after_stage": tuple(unresolved),
            "propagated_inputs": tuple(propagated.columns) if propagated is not None else tuple(),
            "local_aicc": candidate.local_aicc if candidate else np.nan,
            "global_aicc": candidate.global_aicc if candidate else reference_global_aicc,
            "parameter_count": candidate.parameter_count if candidate else 0,
        }
