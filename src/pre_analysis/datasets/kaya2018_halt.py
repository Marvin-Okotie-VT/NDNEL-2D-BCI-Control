"""Kaya et al. 2018, HaLT paradigm. Manual download from Figshare
(MOABB's Kaya2018 only loads the CLA files).

19 EEG (10-20 with old T3/T4/T5/T6 names), 200 Hz, .mat.
Trial: 1 s cue (imagery during it), 1.5-2.5 s random gap.
Passive state = cued rest.
"""

from .spec import DATA_ROOT, DatasetSpec, Run

FILES = {}  # subject int -> filename; fill from the Figshare HaLT listing

MARKERS = {1: "left_hand", 2: "right_hand", 3: "rest",
           4: "feet", 5: "tongue", 6: "feet"}  # 4/6 = left/right leg, pooled
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def load(subject):
    """Return a single Run.

    1. scipy.io.loadmat(path, simplify_cells=True)["o"] -> dict with
       data (n_samples, n_ch), marker (n_samples,), sampFreq, chnames.
    2. marker holds its code for the whole 1 s cue, so onsets are rising
       edges: np.flatnonzero((m[1:] != 0) & (m[:-1] == 0)) + 1.
       Ignore codes > 6 (session markers 91/92/99).
    3. mne.io.RawArray(data.T * scale, info): check the units, MNE wants volts.
       Drop A1, A2 and the X3 sync channel.
    4. raw.rename_channels(RENAME)
    """
    raise NotImplementedError


SPEC = DatasetSpec(
    name="Kaya2018_HaLT",
    subjects=tuple(FILES),
    classes=("left_hand", "right_hand", "feet", "tongue", "rest"),
    rest="cued",
    task_window=(0.0, 1.0),
    rest_window=None,
    load=load,
)