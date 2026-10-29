"""Data checks for the IIIa loader"""

import numpy as np
import pytest

from pre_analysis.data import load_epochs
from pre_analysis.datasets import bci3_iiia
from pre_analysis.datasets.spec import DATA_ROOT

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "bci3_iiia" / "k3b.gdf").exists(),
    reason="IIIa files not downloaded (run pytest from the repo root)",
)

# Trials left after dropping expert-marked artifact trials (1023).
KEPT = {1: 298, 2: 175, 3: 167}


@pytest.mark.parametrize("subject", [1, 2, 3])
def test_loader(subject):
    (run,) = bci3_iiia.load(subject)
    assert len(run.raw.ch_names) == 60
    assert {"C3", "Cz", "C4"} <= set(run.raw.ch_names)
    assert len(run.events) == KEPT[subject]
    assert set(np.unique(run.events[:, 2])) == {1, 2, 3, 4}


def test_tongue_vs_rest_epochs():
    ep = load_epochs("BCI3_IIIa", 1, ("tongue", "rest"))
    assert len(ep["tongue"]) == 74
    assert len(ep["rest"]) == KEPT[1]     # one pre-cue rest epoch per kept trial
    assert ep.get_data().shape[1:] == (60, 501)