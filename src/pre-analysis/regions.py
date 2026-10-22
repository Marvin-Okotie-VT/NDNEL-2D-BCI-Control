"""Named channel groups for region-level comparisons.

Groups list 10-10 labels. A dataset only contributes the channels it has,
so always intersect with epochs.ch_names before use (see channels_in).
"""

REGIONS = {
    "central": ("C3", "Cz", "C4"),
    # T7/T8 sit over temporalis muscle: highest jaw-EMG risk.
    "lateral": ("FC5", "FC6", "C5", "C6", "CP5", "CP6", "T7", "T8"),
}


def channels_in(region, ch_names):
    """Return the channels in a region that are present in ch_names."""
    raise NotImplementedError
