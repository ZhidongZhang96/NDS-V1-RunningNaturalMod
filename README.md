# Running-Speed Modulation of V1 Responses — NDS Visual Coding Project

Final project for **Neural Data Science 2026** (Prof. Philipp Berens), GTC, Tübingen.

**Group members:** **[@Zhidong Zhang](https://github.com/ZhidongZhang96)**, **[@Bach Nguyen](https://github.com/bachnguyenTE)** and **[@Yuzhe Han](https://github.com/Hzzz12138)**.

## What this project studies

It has been shown that layer 2/3 and 4 neurons in mouse V1 are positively modulated by locomotion during visual-responsiveness tasks, specifically for drifting gratings. Using the **Allen Brain Observatory** two-photon calcium-imaging dataset, we ask:

> How does running speed modulate V1 neuron responses, and how does that modulation differ across stimulus types (`drifting_gratings`, `static_gratings`, `natural_scenes`) and compared to spontaneous (no-stimulus) activity?

We focus on how running speed modulates response amplitude, via three analyses:

1. **Binned speed tuning** — bin running speed, build per-neuron tuning curves, test significance with a one-way ANOVA across speed bins, and quantify monotonicity with Spearman's ρ.
2. **Binary running/still conditions** — split trials into *running* vs *still*, compute a sign-safe Modulation Index `MI = (R_run − R_still) / (|R_run| + |R_still| + ε)`, and fit a gain model `R_run = a·R_still + b` (a = multiplicative, b = additive).
3. **Predictive encoding models** — nested ridge-regularized linear models (Null / Add-only / Mult-only / Full) with a tent-basis drifting baseline, compared via cross-validated ΔR² (blocked/purged CV).

> **Status:** all three analyses are implemented in [`utils.py`](utils.py): `SpeedTuning` (Analysis 1) and `BinaryModulation` (Analysis 2) came from the `dev` line; `EncodingModel` (Analysis 3) plus `tests/test_encoding.py` and the `scripts/` pipeline came from the `encoding-model` line. This branch is their integration (all three runnable end-to-end). Reports: [`doc/Modulation&Tuning.md`](doc/Modulation&Tuning.md) (Analyses 1–2), [`doc/EncodingModel.md`](doc/EncodingModel.md) (Analysis 3).
>
> ⚠️ **Cohort caveat (area).** The bundled 47-cell cohort (`data/visual_coding_data.npz`) is Allen container **`511510753` = VISpm**, a higher visual area, *not* V1/VISp, though the layer/line are as intended (Cux2-CreERT2, L2/3–4). **V1** results use a pooled 3-container VISp cohort built via `scripts/download_container.py`. See [`doc/EncodingModel.md`](doc/EncodingModel.md) §9.

## Repository layout

```
NDS-V1-RunningNaturalMod/
├── utils.py                  # Shared code: load_data, extract_trials, TrialData, Plotter + the 3 analysis classes
├── Unified_V1_notebook.ipynb     # ⭐ Start here: demos of all 3 analyses (V1)
├── visual_coding.ipynb       # Data tour + trial-extraction walkthrough
├── Session B&A*.ipynb        # Detailed Analyses 1 & 2 (speed tuning + running modulation), incl. V1 pooling
├── EncodingModel_demo.ipynb  # Detailed Analysis 3 (encoding model, two-metric result)
├── allensdk.ipynb            # Supplementary: pull per-cell metadata via the Allen SDK
├── scripts/                  # Reproducibility pipeline (fetch data, fit, robustness battery, render figures)
├── tests/                    # test_encoding.py, EncodingModel unit tests (synthetic fixture, no data needed)
├── doc/                      # Reports + analysis plan (see "Docs worth reading"); figures under doc/figures/
├── data/                     # Bundled data + committed result caches (see note below)
├── boc/                      # Allen SDK cache manifest + stimulus mappings
├── environment.yml,          # Conda env spec
│   requirements.txt          # pip-level deps (for an existing Python 3.10 env)
└── CLAUDE.md                 # Repo guide for the Claude Code assistant
```

### Core notebooks

| Notebook | What it demonstrates |
|---|---|
| [`Unified_V1_notebook.ipynb`](Unified_V1_notebook.ipynb) | ⭐ **Start here.** One executed notebook with the key result of all three analyses, with **V1 as the main population**, VISpm as a control. |
| [`visual_coding.ipynb`](visual_coding.ipynb) | Data tour + preprocessing: how trials are extracted from the Allen `.npz`. |
| [`Session B&A.ipynb`](Session%20B%26A.ipynb) & [`Session B&A for V1_all.ipynb`](Session%20B%26A%20for%20V1_all.ipynb) | Full **Analyses 1 & 2** (speed tuning + binary running modulation), including the pooled-V1 run. |
| [`EncodingModel_demo.ipynb`](EncodingModel_demo.ipynb) | Full **Analysis 3** (nested encoding model, ΔR² decomposition, two-metric result). |
| [`allensdk.ipynb`](allensdk.ipynb) | Supplementary: pulls per-cell Allen metadata (tuning, reliability, run-modulation, RFs). |

### Docs worth reading

| Doc | What it covers |
|---|---|
| [`doc/Plan.md`](doc/Plan.md) | The analysis plan with the full math for all three analyses. |
| [`doc/Modulation&Tuning.md`](doc/Modulation%26Tuning.md) | **Analyses 1 & 2** report: speed-tuning and running-modulation results. |
| [`doc/EncodingModel.md`](doc/EncodingModel.md) | **Analysis 3** report: the two-metric V1 result, the CV-leakage fix, and area-specificity. |
| [`doc/REFERENCES.md`](doc/REFERENCES.md) | Literature map (cite-X-for-Y) behind the methods and findings. |

> **`data/`** holds the bundled `visual_coding_data.npz` (the VISpm 47-cell cohort) + `neurons_metadata.csv`, plus the committed encoding-model result caches (`encoding_*.npz`, `robust_null.npz`). The large per-container V1 session dumps (`container_*.npz`, ~0.6 GB each) are **fetched on demand** via `scripts/download_container.py` and are git-ignored — `Unified_V1_notebook.ipynb`'s setup cell prints the exact command if they're missing.

## Getting started

### 1. Create the environment

Requires a working [conda](https://docs.conda.io/) (Anaconda/Miniconda). **Recommended — one command from `environment.yml`:**

```bash
conda env create -f environment.yml
conda activate allensdk
```

<details>
<summary>Or set it up manually (equivalent to what <code>environment.yml</code> does)</summary>

```bash
conda create -n allensdk python=3.10 -y
conda activate allensdk
conda install -c conda-forge hdf5 pytables -y   # compiled HDF5 stack
pip install -r requirements.txt                 # allensdk + analysis deps
```
</details>

### 2. Register the Jupyter kernel

So the notebooks can select this environment:

```bash
python -m ipykernel install --user --name allensdk \
  --display-name "Python (allensdk — NDS Visual Coding)"
```

### 3. Get the data

The main notebook loads `visual_coding_data.npz` via `load_data(path="../data")` — it expects a `data/` directory **next to the repo folder**:

```
parent/
├── NDS-V1-RunningNaturalMod/   # this repo
└── data/
    └── visual_coding_data.npz  # NOT included in the repo, obtain separately
```

This `.npz` is git-ignored and must be supplied separately. The supplementary [`allensdk.ipynb`](allensdk.ipynb) does **not** need it — it reads cell metadata via the Allen SDK using the local `boc/` cache.

### 4. Run

```bash
jupyter lab            # or: jupyter notebook
```

Open [`visual_coding.ipynb`](visual_coding.ipynb) and select the **"Python (allensdk — NDS Visual Coding)"** kernel.

## Environment notes

- **Python must be ≥ 3.10** (not 3.9): `utils.py` uses `str | None` runtime unions in the `TrialData` dataclass.
- **`setuptools` is pinned `< 81`**: AllenSDK still imports `pkg_resources`, which newer setuptools removed.
- **`hdf5` + `pytables` come from conda-forge**, not pip (pip's `tables` would need to compile against HDF5).
- AllenSDK 2.16.2 pins older cores: `numpy` 1.23.5, `pandas` 1.5.3, `scipy` 1.10.1.

## Development

- Branches: `dev` (work in progress), `main` (stable).
- The notebook auto-reloads `utils.py` via `%autoreload 2`, so edits to `utils.py` take effect without restarting the kernel.
