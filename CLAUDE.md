# Neural Data Science — Visual Coding Project

## Team
- Zhidong ZHANG, Yuzhe Han, Bach Nguyen
- Summer term 2026, Neural Data Science (Prof. Philipp Berens)

## Research Question
How does running speed modulate V1 neuron responses, and how does this modulation differ across stimulus types (`drifting_gratings`, `static_gratings`, `natural_scenes`) and compared to spontaneous activity?

## Project Structure

```
Projects/
├── utils.py          # Shared code: data loading, plotting, analysis classes
├── utils_bp.py       # Backup of previous procedural version
├── visual_coding.ipynb  # Main notebook (uses utils.py)
├── Plan.md           # Analysis plan with math
└── boc/              # AllenSDK data (supplementary)
```

### utils.py Architecture

- **`load_data()`** — Loads `.npz` into nested `data` dict
- **`Plotter`** — Raw data summary plots (FOV, traces, running speed). Accepts `data` dict
- **`TrialData`** — Dataclass holding extracted trial arrays `(n_trials, n_cells, len_window)`. Does NOT store averages — each analysis class averages as needed
- **`extract_trials()`** — Shared pre-processing: slices ΔF/F and running speed at stim table intervals. Returns `TrialData`. For `stimulus="spontaneous"`, epochs are sourced from `stim_epoch_table` and sliced into fixed-duration windows
- **`SpeedTuning`** — Analysis 1: bin running speed → tuning curve → shuffle test → Spearman
- **`BinaryModulation`** — Analysis 2: running/still classification → MI → gain model
- **`EncodingModel`** — Analysis 3: nested linear models (null/add/mult/full) → R² decomposition

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
