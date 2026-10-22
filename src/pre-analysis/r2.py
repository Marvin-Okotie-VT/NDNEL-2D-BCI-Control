"""Signed r² discriminability maps between two conditions.

Pipeline: epochs -> log power spectrum per epoch -> per (channel, frequency)
correlation with a binary label -> signed r², p-values, rankings.
"""


def log_power(epochs, fmin=1.0, fmax=40.0):
    """Per-epoch log power spectral density.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_freqs)
        log10 PSD. Log makes power roughly Gaussian across trials, which the
        correlation and its t-test assume.
    freqs : ndarray, shape (n_freqs,)

    Notes
    -----
    Welch via epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax,
    n_fft=sfreq, n_per_seg=sfreq) gives 1 Hz bins on a 2 s window: two
    1 s segments averaged, trading frequency resolution for lower variance.
    All epochs must have the same length or the bins will not line up.
    """
    raise NotImplementedError


def signed_r2(X, y):
    """Signed squared point-biserial correlation for every feature.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, ...)
        Features (e.g. log power), any trailing shape.
    y : ndarray of {0, 1}, shape (n_epochs,)
        1 = condition A, 0 = condition B.

    Returns
    -------
    r2 : ndarray, shape X.shape[1:]
        sign(r) * r**2.

    Notes
    -----
    Point-biserial r is Pearson r with a binary variable, which reduces to

        r = (m1 - m0) / s * sqrt(n1 * n0) / n

    m1, m0: class means; n1, n0, n: class and total counts;
    s: population std of X over all epochs (ddof=0).
    Fully vectorized over trailing axes; no loop over channels.
    """
    raise NotImplementedError


def r2_pvalues(r2, n, correction="fdr"):
    """Two-sided p-values for each r², with multiple-comparison correction.

    Notes
    -----
    r is tested with t = r * sqrt((n - 2) / (1 - r**2)), df = n - 2, which is
    equivalent to an independent two-sample t-test. Correct across all
    channel x frequency bins (mne.stats.fdr_correction).

    For task vs pre-cue rest from the same trials the samples are paired;
    this test ignores the pairing and is conservative for the map. Use a
    paired test if a single bin's significance matters.
    """
    raise NotImplementedError


def rank_channels(r2, ch_names, freqs, band=(8.0, 30.0)):
    """Rank channels by peak |r²| inside `band`.

    Returns a list of (channel, peak_r2, peak_freq), sorted descending by
    |peak_r2|. Peak rather than mean, because the reactive band is
    subject-specific and a mean over 8-30 Hz dilutes a narrow mu peak.
    """
    raise NotImplementedError


def region_summary(r2, ch_names, freqs, regions, band=(8.0, 30.0)):
    """Per-region summary of |r²| inside `band`.

    Returns {region: (best_channel, peak_r2, mean_peak_r2_over_channels)}.
    `regions` is preprocessing.regions.REGIONS (or a subset).
    """
    raise NotImplementedError
