"""BCI Competition IV 2a via MOABB.

9 subjects, 22 EEG + 3 EOG, 250 Hz, 2 sessions x 6 runs.
Trial: fixation + beep at 0 s, cue at 2 s, imagery until 6 s.
MOABB marks TRIAL START (dataset.interval == [2, 6]); load() shifts events
by CUE_OFFSET so they mark cue onset.
"""

from .spec import DatasetSpec, Run

CUE_OFFSET = 2.0  # s from MOABB event to cue


def load(subject):
    """Return one Run per MOABB run.

    1. moabb.datasets.BNCI2014_001().get_data([subject])
       -> {subject: {session: {run: Raw}}}; walk both levels.
    2. events = mne.find_events(raw, stim_channel=...) (check raw.ch_names
       for the stim channel name); codes per dataset.event_id.
    3. events[:, 0] += round(CUE_OFFSET * raw.info["sfreq"])
    4. raw.pick("eeg") drops EOG and stim.
    5. MOABB event names already equal canonical labels.
    """
    raise NotImplementedError


SPEC = DatasetSpec(
    name="BNCI2014_001",
    subjects=tuple(range(1, 10)),
    classes=("left_hand", "right_hand", "feet", "tongue"),
    rest="precue",
    task_window=(0.5, 2.5),   # skip the visual-cue response
    rest_window=(-2.0, 0.0),  # fixation; starts at the beep
    load=load,
)