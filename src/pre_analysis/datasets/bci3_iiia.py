"""BCI Competition III dataset IIIa (Graz). Manual download.

3 subjects, 60 EEG, 250 Hz, GDF, one continuous file per subject
(k3b: 9 runs / 360 trials; k6b, l1b: 6 runs / 240 trials; 4 balanced classes).
Trial: blank 0-2 s, beep + cross at 2 s, cue at 3 s, imagery until 7 s.

Files in DATA_ROOT / "bci3_iiia":
    k3b.gdf, k6b.gdf, l1b.gdf
    true_label_k3.txt, true_label_k6.txt, true_label_l1.txt

Verified against the files (2026-10-28):
- GDF annotations: 768 trial start, 785 beep, 786 cross, 769-772 cue
  (training trials), 783 cue (test trials), 1023 expert-marked artifact trial.
  Every cue sits exactly 3 s after its trial start.
- The true-label files list every trial in order (training + test), values
  1-4 in 769-772 order. They agree 100% with the training cues in the GDF.
- No NaNs as read by MNE; data in volts (std ~1e-5). Saturation: 7 samples
  on channel 1 of l1b, outside any trial.
- Channels are stored as "#  1" ... "# 60". The montage (desc Figure 2) is a
  dense custom grid, not 10-10: only C3 (28), Cz (31) and C4 (34) are
  labelled. All other channels keep their number as "E01".."E60".
"""

import mne
import numpy as np

from .spec import DATA_ROOT, DatasetSpec, Run

FILES = {1: "k3b", 2: "k6b", 3: "l1b"}
LABEL_FILES = {1: "true_label_k3.txt", 2: "true_label_k6.txt", 3: "true_label_l1.txt"}

TRIAL_START = 768
CUE_CODES = (769, 770, 771, 772, 783)   # 783 = test trial, class from the label file
ARTIFACT = 1023                         # expert artifact mark, at the trial-start sample
TRUE_LABELS = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
CUE_DELAY = 3.0                         # s from trial start to cue

DROP_ARTIFACT_TRIALS = True             # 17% (k3b), 27% (k6b), 30% (l1b) of trials

ANCHORS = {28: "C3", 31: "Cz", 34: "C4"}  # the only named electrodes 

# Native-grid regions, from figure 2 in description: the two outermost electrodes on each
# side of the C row and of the rows directly in front of and behind it.
REGIONS = {
    "central": ("C3", "Cz", "C4"),
    "lateral": ("E17", "E18", "E26", "E27", "E37", "E38",
                "E24", "E25", "E35", "E36", "E44", "E45"),
}


def channel_names(n=60):
    """Native names: E01..E60, with the three labelled anchors renamed."""
    return [ANCHORS.get(i, f"E{i:02d}") for i in range(1, n + 1)]


def load(subject):
    """Return a single Run with cue-onset events labelled 1-4 (TRUE_LABELS)."""
    folder = DATA_ROOT / "bci3_iiia"
    raw = mne.io.read_raw_gdf(folder / f"{FILES[subject]}.gdf",
                              preload=True, verbose="error")
    raw.rename_channels(dict(zip(raw.ch_names, channel_names(len(raw.ch_names)))))

    # Annotations -> integer events. Mapping each description to its own
    # number keeps the GDF codes readable in the events array.
    wanted = (TRIAL_START, ARTIFACT, *CUE_CODES)
    events, _ = mne.events_from_annotations(
        raw, event_id={str(c): c for c in wanted}, verbose="error")

    starts = events[events[:, 2] == TRIAL_START, 0]
    cues = events[np.isin(events[:, 2], CUE_CODES)]
    labels = np.loadtxt(folder / LABEL_FILES[subject], dtype=int)

    if not len(starts) == len(cues) == len(labels):
        raise ValueError(f"{FILES[subject]}: {len(starts)} trial starts, "
                         f"{len(cues)} cues, {len(labels)} labels")
    delay = cues[:, 0] - starts
    if np.any(delay != round(CUE_DELAY * raw.info["sfreq"])):
        raise ValueError(f"{FILES[subject]}: cue not {CUE_DELAY} s after trial start")
    train = cues[:, 2] != 783
    if np.any(cues[train, 2] - TRIAL_START != labels[train]):
        raise ValueError(f"{FILES[subject]}: label file disagrees with training cues")

    keep = np.ones(len(cues), dtype=bool)
    if DROP_ARTIFACT_TRIALS:
        keep = ~np.isin(starts, events[events[:, 2] == ARTIFACT, 0])

    out = np.column_stack([
        cues[keep, 0],
        np.zeros(keep.sum(), dtype=int),
        labels[keep],
    ])
    event_id = {name: code for code, name in TRUE_LABELS.items()}
    return [Run(raw, out, event_id)]


SPEC = DatasetSpec(
    name="BCI3_IIIa",
    subjects=tuple(FILES),
    classes=("left_hand", "right_hand", "feet", "tongue"),
    rest="precue",
    task_window=(0.5, 2.5),
    rest_window=(-3.0, -1.0),  # blank period, before the beep
    load=load,
)