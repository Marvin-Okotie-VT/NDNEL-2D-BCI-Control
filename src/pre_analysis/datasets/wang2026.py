"""Wang et al. 2026 (CMU, He lab) via MOABB. Online 1D/2D cursor control.

39 subjects in 4 groups, 62 EEG (10-05), 1000 Hz, 5 sessions.
Classes: left hand, right hand, both hands, rest (cued). No tongue or feet.
All trials are closed-loop feedback: keep results separate from the
cue-based datasets. Trial-segmented files cropped to 5 s; MOABB drops
shorter trials.
"""

from .spec import DatasetSpec, Run

GROUP = "bci2000_control"  # joint_learning and tactile_control get tactile
                           # stimulation, which itself modulates mu/beta


def load(subject):
    """Return one Run per session.

    1. moabb.datasets.Wang2026(group=GROUP).get_data([subject])
       -> {subject: {session: {run: Raw}}}
    2. Find events (stim channel or annotations: check raw.ch_names and
       raw.annotations). Check where target onset sits relative to the
       event; shift so events mark cue onset if needed.
    3. raw.pick("eeg")
    """
    raise NotImplementedError


SPEC = DatasetSpec(
    name="Wang2026",
    subjects=(),  # fill from Wang2026(group=GROUP).subject_list
    classes=("left_hand", "right_hand", "hands", "rest"),
    rest="cued",
    task_window=(0.5, 2.5),
    rest_window=None,
    load=load,
)