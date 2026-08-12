"""Minimal end-to-end example with a deliberately imperfect FP model."""

import numpy as np
import pandas as pd

from adopt_fpml import ADOPTFPML, CallableFirstPrinciplesModel, SearchConfig
from adopt_fpml.optimizer import Structure


rng = np.random.default_rng(12)
x = np.linspace(-2, 2, 120)
inputs = pd.DataFrame({"feed": x, "ambient": rng.normal(size=len(x))})
truth = pd.DataFrame({"product": 2 * x + 0.25 * x**2})
fp = CallableFirstPrinciplesModel(
    lambda frame, parameters=None: pd.DataFrame({"product": 2 * frame.feed}, index=frame.index),
    ["product"],
)
result = ADOPTFPML(fp).fit(
    inputs,
    truth,
    {"product": 0.08},
    SearchConfig(max_stages=1, structures=(Structure.PARALLEL, Structure.SERIES_1)),
)
print(result.summary())

