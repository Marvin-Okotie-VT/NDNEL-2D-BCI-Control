"""Contract between dataset modules and the generic pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import mne
import numpy as np

CLASSES = ("left_hand", "right_hand", "hands", "feet", "tongue", "rest")

DATA_ROOT = Path("data")  # manual (non-MOABB) downloads; gitignored


class Run(NamedTuple):
    """One continuous recording, as returned by a dataset loader."""
    raw: mne.io.BaseRaw       # EEG channels only, 10-10 names, volts
    events: np.ndarray        # (n_events, 3); sample column = CUE ONSET
    event_id: dict            # canonical label -> integer code in events


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subjects: tuple
    classes: tuple                          # canonical labels with their own cue
    rest: Literal["cued", "precue"]         # cued: rest is a class; precue: cut from fixation
    task_window: tuple                      # (tmin, tmax) s, relative to cue onset
    rest_window: tuple | None               # precue: relative to the task cue (required)
                                            # cued: relative to the rest cue; None = task_window
    load: Callable[[int], list]             # subject -> list[Run]

    def __post_init__(self):
        unknown = set(self.classes) - set(CLASSES)
        if unknown:
            raise ValueError(f"{self.name}: non-canonical classes {unknown}")
        if self.rest == "precue" and self.rest_window is None:
            raise ValueError(f"{self.name}: precue rest needs a rest_window")
        if self.rest_window is not None:
            task_len = self.task_window[1] - self.task_window[0]
            rest_len = self.rest_window[1] - self.rest_window[0]
            if abs(task_len - rest_len) > 1e-9:
                raise ValueError(f"{self.name}: task and rest windows differ in length")