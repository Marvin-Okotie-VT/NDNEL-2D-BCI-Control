"""Synthetic tests for load_epochs: one precue dataset, one cued dataset."""
import mne
import numpy as np
import pytest

from pre_analysis import data
from pre_analysis.datasets.spec import DatasetSpec, Run

SF = 250.0
CH = ["C3", "Cz", "C4", "C5", "C6"]


def make_run(codes, cue_at=3.0, gap=8.0, seed=0):
    """Continuous noise with one event per trial at cue_at + k*gap seconds."""
    rng = np.random.default_rng(seed)
    n_trials = len(codes)
    n_samp = int((cue_at + n_trials * gap) * SF)
    info = mne.create_info(CH, SF, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((len(CH), n_samp)) * 1e-5, info, verbose=False)
    onsets = (np.arange(n_trials) * gap + cue_at) * SF
    events = np.column_stack([onsets.astype(int), np.zeros(n_trials, int), codes])
    return raw, events


def precue_load(subject):
    ev_id = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
    runs = []
    for seed in range(2):
        raw, events = make_run([1, 2, 3, 4] * 3, seed=seed)
        runs.append(Run(raw, events, ev_id))
    return runs


def cued_load(subject):
    # two run types, like PhysionetMI: L/R+rest runs and hands/feet+rest runs
    a_raw, a_ev = make_run([1, 9, 2, 9] * 3, seed=1)
    b_raw, b_ev = make_run([5, 9, 3, 9] * 3, seed=2)
    return [Run(a_raw, a_ev, {"left_hand": 1, "right_hand": 2, "rest": 9}),
            Run(b_raw, b_ev, {"hands": 5, "feet": 3, "rest": 9})]


PRECUE = DatasetSpec("FAKE_PRECUE", (1,), ("left_hand", "right_hand", "feet", "tongue"),
                     "precue", (0.5, 2.5), (-2.0, 0.0), precue_load)
CUED = DatasetSpec("FAKE_CUED", (1,), ("left_hand", "right_hand", "hands", "feet", "rest"),
                   "cued", (0.5, 2.5), (1.5, 3.5), cued_load)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setitem(data.REGISTRY, PRECUE.name, PRECUE)
    monkeypatch.setitem(data.REGISTRY, CUED.name, CUED)


def test_precue_tongue_vs_rest():
    ep = data.load_epochs("FAKE_PRECUE", 1, ("tongue", "rest"))
    # 2 runs x 3 tongue trials; rest from EVERY cue: 2 runs x 12 trials
    assert len(ep["tongue"]) == 6
    assert len(ep["rest"]) == 24
    assert ep.get_data().shape[1:] == (5, 501)       # 2 s at 250 Hz, tmax inclusive
    assert ep.event_id == {"tongue": 5, "rest": 6}   # CLASSES.index + 1


def test_rest_is_pre_cue_signal():
    """The rest epoch must be the 2 s before the cue, not after it."""
    raw, events = precue_load(1)[0].raw, precue_load(1)[0].events
    raw = raw.copy().set_eeg_reference("average", projection=False, verbose=False)
    start = events[0, 0] - int(2.0 * SF)
    expected = raw.get_data()[:, start:start + 501]
    ep = data.load_epochs("FAKE_PRECUE", 1, ("rest",))
    np.testing.assert_allclose(ep.get_data()[0], expected)


def test_cued_across_run_types():
    ep = data.load_epochs("FAKE_CUED", 1, ("feet", "hands", "rest"))
    assert len(ep["feet"]) == 3 and len(ep["hands"]) == 3
    assert len(ep["rest"]) == 12


def test_errors():
    with pytest.raises(ValueError, match="unknown dataset"):
        data.load_epochs("NOPE", 1, ("tongue",))
    with pytest.raises(ValueError, match="no 'tongue'"):
        data.load_epochs("FAKE_CUED", 1, ("tongue",))
    with pytest.raises(ValueError, match="not canonical"):
        data.load_epochs("FAKE_PRECUE", 1, ("tounge",))
    with pytest.raises(ValueError, match="duplicate"):
        data.load_epochs("FAKE_PRECUE", 1, ("tongue", "tongue"))


def test_filter_runs():
    ep = data.load_epochs("FAKE_PRECUE", 1, ("tongue", "rest"), l_freq=1.0, h_freq=40.0)
    assert len(ep) == 30