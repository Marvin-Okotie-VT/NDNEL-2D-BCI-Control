"""Compute r2 maps for one contrast across subjects.

Example
-------
python scripts/run_r2_map.py --dataset BNCI2014_001 --subjects 1 2 3 \
    --class-a tongue --class-b rest

Writes to results/r2_map/<dataset>/<class_a>_vs_<class_b>/:
    sub-XX_r2.npz      signed r2, p-values, ch_names, freqs
    sub-XX_heatmap.png channel x frequency
    sub-XX_topo.png    topomap of mean r2 over the mu/beta band
    ranking.csv        per-subject channel ranking + region summary
    pooled_*.png       across-subject mean map
"""