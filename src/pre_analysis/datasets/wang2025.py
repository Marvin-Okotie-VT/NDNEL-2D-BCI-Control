"""Wang, Yao & Wang 2025 (ZJU-MI-EEG, MI4 subset) via MOABB.

15 subjects, 62 EEG (10-10), 256 Hz, 2 days x (calibration + feedback) run.
Classes: left hand, right hand, tongue, both feet. No rest class.
Released as trials of -1 to 4 s around the cue, concatenated with
"EDGE boundary" annotations; MOABB's stim events sit at cue onset.
"""

from .spec import DatasetSpec, Run

RUNS = ("0calibration",)  # open-loop only; "1feedback" is closed-loop


def load(subject):
    """Return one Run per calibration run (one per day).

    1. moabb.datasets.Wang2025().get_data([subject])
       -> {subject: {session: {run: Raw}}}; keep run keys in RUNS.
    2. events = mne.find_events(raw, stim_channel="STI 014"); codes per
       dataset.event_id (left_hand 1, right_hand 2, tongue 3, feet 4).
       Already at cue onset: no shift.
    3. raw.pick("eeg")
    4. Check in the paper what was on screen from -1 to 0 s: precue rest
       assumes a fixation period.
    """
    raise NotImplementedError


SPEC = DatasetSpec(
    name="Wang2025",
    subjects=tuple(range(1, 16)),
    classes=("left_hand", "right_hand", "feet", "tongue"),
    rest="precue",
    task_window=(0.5, 1.5),   # 1 s: limited by the 1 s of pre-cue data
    rest_window=(-1.0, 0.0),
    load=load,
)