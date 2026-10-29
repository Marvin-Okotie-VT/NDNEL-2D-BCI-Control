"""Generic pipeline: any registered dataset -> epochs with canonical labels.

Dataset-specific work (file formats, event codes, cue alignment, channel
names) happens in the loaders under datasets/. Everything here is the same
for every dataset and relies only on the Run contract in datasets/spec.py.
"""

import mne
import numpy as np

from .datasets import REGISTRY
from .datasets.spec import CLASSES 


def load_epochs(dataset, subject, classes, l_freq=None, h_freq=None):
    """Epochs for the requested canonical classes, identical steps for every dataset.

    Parameters
    ----------
    dataset : str
        Key of REGISTRY, e.g. "BNCI2014_001".
    subject : int
    classes : sequence of str
        Canonical labels, e.g. ("tongue", "rest").
    l_freq, h_freq : float | None
        Optional band-pass on the continuous data. None (default) skips it:
        Welch already separates frequencies, and the trial-segmented datasets
        (Wang2025, Wang2026) ring at segment edges when filtered. raw.filter's
        default skip_by_annotation=("edge", ...) filters each "EDGE boundary"
        segment on its own.

    Returns
    -------
    mne.EpochsArray
        All epochs have the same length and start at t = 0. event_id maps each
        requested label to CLASSES.index(label) + 1, so codes are identical
        across datasets and calls.
    """
    spec = _get_spec(dataset)
    classes = _check_classes(spec, classes)
    task_labels = [c for c in classes if c != "rest"]
    want_rest = "rest" in classes

    runs = spec.load(subject)
    if not runs:
        raise ValueError(f"{dataset} subject {subject}: loader returned no runs")

    blocks, labels, info = [], [], None
    for i, run in enumerate(runs):
        raw = run.raw.load_data()
        if l_freq is not None or h_freq is not None:
            raw.filter(l_freq, h_freq, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)

        if info is None:
            info = raw.info
        elif raw.ch_names != info.ch_names or raw.info["sfreq"] != info["sfreq"]:
            raise ValueError(
                f"{dataset} subject {subject}: run {i} differs from run 0 in "
                "channels or sampling rate"
            )

        code_to_label = {code: label for label, code in run.event_id.items()}

        # Task epochs: only the requested classes this run actually contains.
        task_codes = [run.event_id[c] for c in task_labels if c in run.event_id]
        X, codes = _epoch(raw, _select(run.events, task_codes), spec.task_window)
        blocks.append(X)
        labels += [code_to_label[c] for c in codes]

        if want_rest:
            if spec.rest == "cued":
                rest_codes = [run.event_id["rest"]] if "rest" in run.event_id else []
                window = spec.rest_window if spec.rest_window is not None else spec.task_window
            else:  # precue: one rest epoch before EVERY task cue, any class
                rest_codes = [code for label, code in run.event_id.items() if label != "rest"]
                window = spec.rest_window
            X, _ = _epoch(raw, _select(run.events, rest_codes), window)
            blocks.append(X)
            labels += ["rest"] * len(X)

    lengths = {b.shape[-1] for b in blocks if len(b)}
    if len(lengths) > 1:
        raise ValueError(
            f"{dataset}: epoch lengths differ {sorted(lengths)} samples; "
            "task/rest windows x sfreq must land on whole samples"
        )

    counts = {c: labels.count(c) for c in classes}
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise ValueError(f"{dataset} subject {subject}: no epochs for {missing}")

    data = np.concatenate([b for b in blocks if len(b)])
    event_id = {c: CLASSES.index(c) + 1 for c in classes}
    n, n_times = len(data), data.shape[-1]
    # Column 0 is a nominal sample index; it only has to be unique and increasing.
    events = np.column_stack([
        np.arange(n) * n_times,
        np.zeros(n, dtype=int),
        [event_id[label] for label in labels],
    ])
    return mne.EpochsArray(data, info, events=events, tmin=0.0,
                           event_id=event_id, verbose=False)


def _get_spec(dataset):
    if dataset not in REGISTRY:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(REGISTRY)}")
    return REGISTRY[dataset]


def _check_classes(spec, classes):
    """Validate requested labels against CLASSES and this dataset; return a tuple."""
    classes = tuple(classes)
    if not classes:
        raise ValueError("request at least one class")
    if len(set(classes)) != len(classes):
        raise ValueError(f"duplicate classes in {classes}")
    for c in classes:
        if c not in CLASSES:
            raise ValueError(f"{c!r} is not canonical; choose from {CLASSES}")
        available = c in spec.classes or (c == "rest" and spec.rest == "precue")
        if not available:
            raise ValueError(f"{spec.name} has no {c!r} trials")
    return classes


def _select(events, codes):
    """Rows of `events` whose code (column 2) is in `codes`."""
    return events[np.isin(events[:, 2], codes)]


def _epoch(raw, events, window):
    """Cut `window` (s, relative to each event) around `events`.

    Returns (data, codes): data (n_kept, n_channels, n_times), codes (n_kept,).
    MNE drops epochs that run past the recording or overlap BAD annotations
    (e.g. the "BAD boundary" between concatenated runs), so n_kept can be
    smaller than len(events).
    """
    if len(events) == 0:
        return np.empty((0, len(raw.ch_names), 0)), np.empty(0, dtype=int)

    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=window[0], tmax=window[1],
        baseline=None, preload=True, verbose=False,
    )
    data = epochs.get_data()
    if np.isnan(data).any():
        raise ValueError("NaNs in epoch data; the loader must remove or interpolate them")
    return data, epochs.events[:, 2]