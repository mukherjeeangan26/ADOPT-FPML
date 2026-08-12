import numpy as np
import pandas as pd

from adopt_fpml import ADOPTFPML, CallableFirstPrinciplesModel, MLTrainingConfig, NumpyMLPTrainer
from adopt_fpml.optimizer import SearchConfig, Structure


def test_generic_optimizer_discovers_parallel_correction():
    x = np.linspace(-1, 1, 80)
    inputs = pd.DataFrame({"x": x, "unused": np.sin(13 * x)})
    outputs = pd.DataFrame({"y": 3 * x + 0.2})
    fp = CallableFirstPrinciplesModel(
        lambda frame, parameters=None: pd.DataFrame({"y": np.zeros(len(frame))}, index=frame.index),
        ["y"],
    )
    trainer = NumpyMLPTrainer(MLTrainingConfig(hidden_neurons=6, epochs=600, patience=80, seed=4))
    result = ADOPTFPML(fp, ml_trainer=trainer).fit(
        inputs,
        outputs,
        {"y": 0.12},
        SearchConfig(max_stages=1, structures=(Structure.PARALLEL,), max_global_aicc_relative_increase=0.0),
    )
    assert not result.unresolved_outputs
    assert result.stages[0].structure == Structure.PARALLEL
    assert result.stages[0].input_names[0] == "x"

