"""Compute r² maps for one contrast across subjects.

Example
-------
python scripts/run_r2_map.py --dataset BNCI2014_001 --subjects 1 2 3 \
    --class-a tongue --class-b rest

Writes to results/r2_map/<dataset>/<class_a>_vs_<class_b>/:
    sub-XX_r2.npz      signed r², p-values, ch_names, freqs
    sub-XX_heatmap.png channel x frequency
    sub-XX_topo.png    topomap of mean r² over the mu/beta band
    ranking.csv        per-subject channel ranking + region summary
    pooled_*.png       across-subject mean map
"""

import argparse

from pre_analysis.data import CLASSES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--class-a", required=True, choices=CLASSES)
    p.add_argument("--class-b", required=True, choices=CLASSES)
    p.add_argument("--fmin", type=float, default=1.0)
    p.add_argument("--fmax", type=float, default=40.0)
    p.add_argument("--band", type=float, nargs=2, default=(8.0, 30.0),
                   help="band for ranking and topomaps")
    p.add_argument("--out", default="results/r2_map")
    return p.parse_args()


def main():
    args = parse_args()
    if args.class_a == args.class_b:
        raise SystemExit("class-a and class-b must differ")
    # Per subject:
    #   epochs = load_epochs(dataset, subject, (class_a, class_b))
    #   X, freqs = log_power(epochs, fmin, fmax); y = (label == class_a)
    #   r2 = signed_r2(X, y); p = r2_pvalues(r2, len(y))
    #   save npz, plot heatmap + topomap, rank_channels, region_summary
    # Pooled: average signed r² across subjects; report how many subjects
    #   show a significant effect per channel, not just the mean.
    raise NotImplementedError


if __name__ == "__main__":
    main()