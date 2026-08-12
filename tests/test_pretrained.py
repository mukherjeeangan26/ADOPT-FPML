import pandas as pd

from adopt_fpml.case_studies.e1 import MEASURED_COLUMNS as E1_MEASURED
from adopt_fpml.case_studies.e1 import OUTPUTS as E1_OUTPUTS
from adopt_fpml.case_studies.e1 import run_e1_pretrained
from adopt_fpml.case_studies.e2 import MEASURED_COLUMNS as E2_MEASURED
from adopt_fpml.case_studies.e2 import OUTPUTS as E2_OUTPUTS
from adopt_fpml.case_studies.e2 import run_e2_pretrained
from adopt_fpml.datasets import load_dataset
from adopt_fpml.metrics import rmse_by_output


def _rmse(case, runner, measured, outputs):
    data = load_dataset(case)
    truth = data[measured].copy()
    truth.columns = outputs
    return pd.Series(rmse_by_output(runner(data), truth))


def test_e1_training_reproduction_matches_reported_values():
    rmse = _rmse("e1_training", run_e1_pretrained, E1_MEASURED, E1_OUTPUTS)
    assert (rmse - pd.Series({"CA": 2.9, "CB": 1.5, "CC": 1.4, "T_out": 6.4})).abs().max() < 0.06


def test_e1_heldout_reproduction_matches_reported_values():
    data = load_dataset("e1_continuous")
    truth = data[E1_MEASURED].copy()
    truth.columns = E1_OUTPUTS
    rmse = pd.Series(rmse_by_output(run_e1_pretrained(data).iloc[500:], truth.iloc[500:]))
    expected = pd.Series({"CA": 7.3, "CB": 2.8, "CC": 3.0, "T_out": 19.0})
    assert (rmse - expected).abs().max() < 0.06


def test_e2_training_and_testing_reproduction_matches_reported_values():
    train = _rmse("e2_training", run_e2_pretrained, E2_MEASURED, E2_OUTPUTS)
    test = _rmse("e2_test", run_e2_pretrained, E2_MEASURED, E2_OUTPUTS)
    assert abs(train["F_liq"] - 5.1e-4) < 1e-5
    assert abs(train["T_vap"] - 7.1) < 0.02
    assert abs(test["y_A"] - 1.0e-3) < 5e-5
    assert abs(test["T_liq"] - 7.2) < 0.02
