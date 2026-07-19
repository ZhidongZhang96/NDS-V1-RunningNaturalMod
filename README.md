# Running-Speed Modulation of V1 Responses — NDS Visual Coding Project

Final project for **Neural Data Science 2026** (Prof. Philipp Berens), GTC, Tübingen.

**Group members:** **[@Zhidong Zhang](https://github.com/ZhidongZhang96)**, **[@Yuzhe Han](https://github.com/Hzzz12138)** and **[@Bach Nguyen](https://github.com/bachnguyenTE)**.

## What this project studies

It has been shown that layer 2/3 and 4 neurons in mouse V1 are positively modulated by locomotion during visual-responsiveness tasks, specifically for drifting gratings. Using the **Allen Brain Observatory** two-photon calcium-imaging dataset, we ask:

> How does running speed modulate V1 neuron responses, and how does that modulation differ across stimulus types (`drifting_gratings`, `static_gratings`, `natural_scenes`) and compared to spontaneous (no-stimulus) activity?

We focus on how running speed modulates response amplitude, via three analyses (see [`Plan.md`](doc/Plan.md) for the full math):

1. **Binned speed tuning** — bin running speed, build per-neuron tuning curves, test significance with a one-way ANOVA across speed bins, and quantify monotonicity with Spearman's ρ.
2. **Binary running/still conditions** — split trials into *running* vs *still*, compute a sign-safe Modulation Index `MI = (R_run − R_still) / (|R_run| + |R_still| + ε)`, and fit a gain model `R_run = a·R_still + b` (a = multiplicative, b = additive).
3. **Predictive encoding models** — nested ridge-regularized linear models (Null / Add-only / Mult-only / Full) with a tent-basis drifting baseline, compared via cross-validated ΔR² (blocked/purged CV).

> **Status:** all three analyses are implemented in [`utils.py`](utils.py) — `SpeedTuning` (Analysis 1) and `BinaryModulation` (Analysis 2) came from the `dev` line; `EncodingModel` (Analysis 3) plus `tests/test_encoding.py` and the `scripts/` pipeline came from the `encoding-model` line. This branch is their integration (all three runnable end-to-end). Reports: [`doc/Modulation&Tuning.md`](doc/Modulation&Tuning.md) (Analyses 1–2), [`doc/EncodingModel.md`](doc/EncodingModel.md) (Analysis 3).
>
> ⚠️ **Cohort caveat (area).** The bundled 47-cell cohort (`data/visual_coding_data.npz`) is Allen container **`511510753` = VISpm**, a higher visual area — *not* V1/VISp — though the layer/line are as intended (Cux2-CreERT2, L2/3–4). Genuine **V1** results use a pooled 3-container VISp cohort (n=363) built via `scripts/download_container.py`. See [`doc/TEAM_NOTE.md`](doc/TEAM_NOTE.md).

## Repository layout

| Path | What it is |
|---|---|
| [`utils.py`](utils.py) | Shared code: `load_data`, `extract_trials`, `TrialData`, `Plotter`, and the 3 analysis classes |
| [`visual_coding.ipynb`](visual_coding.ipynb) | Main analysis notebook (loads the exported `.npz`, uses `utils.py`) |
| [`allensdk.ipynb`](allensdk.ipynb) | Supplementary: pulls cell metadata directly from the Allen SDK |
| [`neurons_metadata.csv`](neurons_metadata.csv) | Exported per-cell Allen metrics (tuning, reliability, run-modulation, receptive fields) |
| [`Plan.md`](doc/Plan.md) | Analysis plan with the mathematical details |
| `boc/` | Allen SDK cache manifest + stimulus mappings |
| `environment.yml`, `requirements.txt` | Environment specs (see below) |

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
    └── visual_coding_data.npz  # NOT included in the repo — obtain separately
```

This `.npz` is git-ignored and must be supplied separately. The supplementary [`allensdk.ipynb`](allensdk.ipynb) does **not** need it — it reads cell metadata via the Allen SDK using the local `boc/` cache.

### 4. Run

```bash
jupyter lab            # or: jupyter notebook
```

Open [`visual_coding.ipynb`](visual_coding.ipynb) and select the **"Python (allensdk — NDS Visual Coding)"** kernel.

## Environment notes / gotchas

These are deliberate and load-bearing — don't "simplify" them away:

- **Python must be ≥ 3.10** (not 3.9): `utils.py` uses `str | None` runtime unions in the `TrialData` dataclass.
- **`setuptools` is pinned `< 81`**: AllenSDK still imports `pkg_resources`, which newer setuptools removed.
- **`hdf5` + `pytables` come from conda-forge**, not pip (pip's `tables` would need to compile against HDF5).
- AllenSDK 2.16.2 pins older cores: `numpy` 1.23.5, `pandas` 1.5.3, `scipy` 1.10.1.

## Development

- Branches: `dev` (work in progress), `master` (stable).
- The notebook auto-reloads `utils.py` via `%autoreload 2`, so edits to `utils.py` take effect without restarting the kernel.
