import numpy as np
import pandas as pd

from adopt_fpml.metrics import MetricThreshold, aicc, rmse_by_output


def test_rmse_and_thresholds():
    true = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 1.0]})
    pred = pd.DataFrame({"a": [0.0, 0.0], "b": [1.0, 3.0]})
    assert rmse_by_output(pred, true) == {"a": np.sqrt(2), "b": np.sqrt(2)}
    assert MetricThreshold(1.5).evaluate(pred.a, true.a)[0]
    assert not MetricThreshold(1.0).evaluate(pred.a, true.a)[0]


def test_aicc_small_sample_guard_and_finite_value():
    true = np.arange(20.0).reshape(10, 2)
    pred = true + 0.1
    assert np.isfinite(aicc(pred, true, 3))
    assert np.isinf(aicc(pred[:1], true[:1], 1))

