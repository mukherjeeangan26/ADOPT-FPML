import numpy as np
import importlib.util

from adopt_fpml.case_studies.e1 import (
    E1FirstPrinciplesModel,
    E1ParameterTargetGenerator,
    INPUTS as E1_INPUTS,
    MEASURED_COLUMNS as E1_MEASURED,
    OUTPUTS as E1_OUTPUTS,
    generate_e1_data,
)
from adopt_fpml.case_studies.e2 import (
    E2FirstPrinciplesModel,
    E2ParameterTargetGenerator,
    MEASURED_COLUMNS as E2_MEASURED,
    OUTPUTS as E2_OUTPUTS,
    generate_e2_data,
)
from adopt_fpml.datasets import load_dataset


def test_e1_generator_reproduces_training_data():
    expected = load_dataset("e1_training")
    actual = generate_e1_data(500)
    assert list(actual) == list(expected)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=4e-11)


def test_reconstructed_e2_generator_reproduces_authoritative_data():
    feed = load_dataset("e1_training")
    expected = load_dataset("e2_training")
    actual = generate_e2_data(feed)
    assert list(actual) == list(expected)
    assert actual.phase_state.tolist() == expected.phase_state.tolist()
    numeric = expected.select_dtypes("number").columns
    np.testing.assert_allclose(actual[numeric], expected[numeric], rtol=2e-8, atol=1e-5)


def test_case_study_parameter_estimators_return_finite_targets():
    if importlib.util.find_spec("scipy") is None:
        return  # CI/package installs include required SciPy; the bundled QA runtime does not.
    e1 = load_dataset("e1_training").iloc[:4]
    e1_truth = e1[E1_MEASURED].copy()
    e1_truth.columns = E1_OUTPUTS
    e1_fp_model = E1FirstPrinciplesModel()
    e1_inputs = e1[E1_INPUTS]
    e1_theta = E1ParameterTargetGenerator(e1_fp_model, local_search_steps=2).generate(
        e1_inputs, e1_truth, e1_fp_model.predict(e1_inputs), E1_OUTPUTS
    )
    assert list(e1_theta) == ["K"] and np.isfinite(e1_theta).all().all()

    e2 = load_dataset("e2_training").iloc[:3]
    e2_inputs = e2[["CA_feed_measured", "CB_feed_measured", "CC_feed_measured", "T_feed_measured", "v"]].copy()
    e2_inputs.columns = ["CA_feed", "CB_feed", "CC_feed", "T_feed", "v"]
    e2_truth = e2[E2_MEASURED].copy()
    e2_truth.columns = E2_OUTPUTS
    e2_fp_model = E2FirstPrinciplesModel()
    e2_theta = E2ParameterTargetGenerator(e2_fp_model, local_search_steps=2).generate(
        e2_inputs, e2_truth, e2_fp_model.predict(e2_inputs), ["F_liq", "y_A"]
    )
    assert list(e2_theta) == ["UA", "K_A", "K_B", "K_C"]
    assert np.isfinite(e2_theta).all().all()
