EEG motor-imagery decoding for 2D BCI control.
Neural Dynamics and Neuroengineering Lab, Virginia Tech.
Marvin Okotie

## Layout

```
src/preprocessing/   data loading, cleaning, and validation of design choices
scripts/             runnable analyses
results/             outputs (not tracked)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Raw data is not tracked. Public datasets are fetched through MOABB
(cached in `~/mne_data`).