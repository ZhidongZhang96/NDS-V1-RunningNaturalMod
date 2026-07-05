# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Neural Data Science — Visual Coding Project

## Team
- Zhidong ZHANG, Yuzhe Han, Bach Nguyen
- Summer term 2026, Neural Data Science (Prof. Philipp Berens)

## Research Question
How does running speed modulate V1 neuron responses, and how does this modulation differ across stimulus types (`drifting_gratings`, `static_gratings`, `natural_scenes`) and compared to spontaneous activity?

## Environment Setup

The project runs in a conda environment named `allensdk` (Python **3.10**). See [README.md](README.md) for the full walkthrough. Quickest path:

```bash
conda env create -f environment.yml   # creates env "allensdk"
conda activate allensdk
python -m ipykernel install --user --name allensdk \
  --display-name "Python (allensdk — NDS Visual Coding)"
```

Critical, non-obvious constraints (do not "simplify" these away):
- **Python must be ≥ 3.10.** `utils.py` uses PEP 604 runtime unions (`str | None`) in the `TrialData` dataclass with no `from __future__ import annotations`, so it fails to import on 3.9 — even though `allensdk.ipynb`'s inline setup cell suggests 3.9.
- **`setuptools` must be < 81.** AllenSDK imports `pkg_resources`, which newer setuptools removed; without the pin `import allensdk...` raises `ModuleNotFoundError: No module named 'pkg_resources'`.
- **`hdf5` + `pytables` come from conda-forge**, not pip (pip's `tables` would need to compile against HDF5 headers). AllenSDK itself is pip-installed.
- AllenSDK pins older cores: `numpy` 1.23.5, `pandas` 1.5.3, `scipy` 1.10.1. Don't upgrade these expecting AllenSDK to keep working.

## Running

- Main analysis: open `visual_coding.ipynb` and select the `allensdk` kernel.
- **The main notebook needs `visual_coding_data.npz`**, loaded via `load_data(path="../data")` — i.e. a `data/` directory *beside* the repo. This file is git-ignored and **not in the repo**; it must be supplied separately before the notebook can run end-to-end.
- `allensdk.ipynb` pulls cell metadata from the Allen SDK using the local `boc/manifest.json` cache and runs without the `.npz`.
- There is no test suite, linter config, or build step in this repo; "running" means executing the notebooks.

## Project Structure

```
NDS-V1-RunningNaturalMod/
├── utils.py             # Shared code: data loading, plotting, analysis classes
├── visual_coding.ipynb  # Main notebook (uses utils.py, loads ../data/visual_coding_data.npz)
├── allensdk.ipynb       # Supplementary: pulls cell metadata via AllenSDK + boc/ cache
├── neurons_metadata.csv # Exported per-cell Allen metrics (tuning, reliability, run-mod, RFs)
├── Plan.md              # Analysis plan with the full math
├── environment.yml      # Conda env spec (see Environment Setup)
├── requirements.txt     # pip-level deps (for an existing Python 3.10 env)
└── boc/                 # AllenSDK cache manifest + stimulus mappings (supplementary)
```

Data (`visual_coding_data.npz`) lives outside the repo in `../data/` and is not tracked.

### utils.py Architecture

- **`load_data()`** — Loads `.npz` into nested `data` dict
- **`Plotter`** — Raw data summary plots (FOV, traces, running speed). Accepts `data` dict
- **`TrialData`** — Dataclass holding extracted trial arrays `(n_trials, n_cells, len_window)`. Does NOT store averages — each analysis class averages as needed
- **`extract_trials()`** — Shared pre-processing: slices ΔF/F and running speed at stim table intervals. Returns `TrialData`. For `stimulus="spontaneous"`, epochs are sourced from `stim_epoch_table` and sliced into fixed-duration windows
- **`SpeedTuning`** — Analysis 1: bin running speed → tuning curve → shuffle test → Spearman
- **`BinaryModulation`** — Analysis 2: running/still classification → MI → gain model
- **`EncodingModel`** — Analysis 3: nested linear models (null/add/mult/full) → R² decomposition

The three analysis classes are currently **stubs** — their `compute_*`/`fit_*`/`plot_*` methods `raise NotImplementedError`. The data-loading, `extract_trials`, and `Plotter` code is implemented. `Plan.md` holds the math each stub should implement.

### Conventions

- Analysis classes store input as `self._td`, compute results into `self.xxx` attributes
- Plotting methods read stored results, do not re-compute
- Each analysis class can be applied to any stimulus type via identical `TrialData` interface
- Non-implemented methods raise `NotImplementedError` (stubs in place)
- `extract_trials()` takes `response_window=(offset, duration)` for stimulus-specific response windows

### Stimulus × Session Mapping

| Stimulus | Session | n_trials |
|---|---|---|
| drifting_gratings | A | 628 |
| static_gratings | B | 6000 |
| natural_scenes | B | 5950 |
| locally_sparse_noise | C | 8880 |

30 Hz sampling rate. dg trials ≈ 60 frames (2 s), sg/ns trials ≈ 7 frames (0.23 s).

## Development

- Branch: `dev` (work in progress), `master` (stable)
- Notebook auto-reloads utils via `%autoreload 2`
