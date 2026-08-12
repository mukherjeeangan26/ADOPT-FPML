import pandas as pd

from adopt_fpml.pruning import PruningConfig
from adopt_fpml.selectors import PearsonSelector, SpearmanSelector


def test_selectors_rank_relevant_input_first():
    candidates = pd.DataFrame({"noise": [1, 0, 1, 0, 1], "signal": [0, 1, 2, 3, 4]})
    targets = pd.DataFrame({"y": [0, 2, 4, 6, 8]})
    assert PearsonSelector().rank(candidates, targets)[0] == "signal"
    assert SpearmanSelector().rank(candidates, targets)[0] == "signal"


def test_pruning_extensions():
    assert PruningConfig().should_stop([10, 9, 9.1])
    assert not PruningConfig(strategy="patience", patience=2).should_stop([10, 9, 9.1, 9.2])
    assert PruningConfig(strategy="patience", patience=2).should_stop([10, 9, 9.1, 9.2, 9.3])
    assert PruningConfig(strategy="moving_window", window=2).should_stop([10, 9, 9.5, 9.6])

