"""PhysioNet EEG Motor Movement/Imagery dataset via MOABB (imagery runs only).

109 subjects, 64 EEG, 160 Hz. Imagery runs 4/8/12: left vs right fist;
runs 6/10/14: both fists vs both feet. ~4 s task periods (T1/T2)
alternate with ~4 s relax periods (T0), which MOABB labels "rest".
No tongue. Left/right and hands/feet come from different runs, so
contrasts across the two run types also compare runs.
"""

from .spec import DatasetSpec, Run

EXCLUDE = {88}  # recorded at 128 Hz (MOABB note)


def load(subject):
    """Return one Run per imagery run.

    1. moabb.datasets.PhysionetMI(imagined=True, executed=False).get_data([subject])
    2. Events come from the T0/T1/T2 annotations at period onset, so they
       already mark cue onset. Check dataset.event_id and that "rest" (T0)
       events are present.
    3. Channel names: raw EDF labels look like "Fc5."; if so,
       mne.datasets.eegbci.standardize(raw).
    4. raw.pick("eeg")
    """
    raise NotImplementedError


SPEC = DatasetSpec(
    name="PhysionetMI",
    subjects=tuple(s for s in range(1, 110) if s not in EXCLUDE),
    classes=("left_hand", "right_hand", "hands", "feet", "rest"),
    rest="cued",
    task_window=(0.5, 2.5),
    rest_window=(1.5, 3.5),  # relative to T0: skips the post-imagery beta
                             # rebound right after the task period ends
    load=load,
)