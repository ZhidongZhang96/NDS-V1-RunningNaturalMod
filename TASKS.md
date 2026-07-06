# TASKS — Dividing the Analysis & Model-Building Work

Work plan for implementing the three analysis classes in [`utils.py`](utils.py) and answering the
research question (how running speed modulates V1 responses, **comparing gratings vs natural
scenes**, across `drifting_gratings`, `static_gratings`, `natural_scenes`, and `spontaneous`).

**Division:** by analysis — each person owns one analysis class end-to-end. The three are
unbalanced (C ≫ A > B), so the two lighter roles carry the shared/integration work.
Map **Person A / B / C** to Zhidong / Yuzhe / Bach among yourselves.

## Reference repositories (port, don't reinvent)

Two published repos are near-exact references for our analyses. **Both are grating-only** — see the
"Grating vs Natural" section for what this means for the natural-scenes half of the project. For the published *literature* (natural-scene coding, running modulation, encoding-model methods) and a cite-X-for-Y map, see [`REFERENCES.md`](REFERENCES.md).

**[achristensen56/AIBSmouselocomotion](https://github.com/achristensen56/AIBSmouselocomotion)** — Christensen et al., Python, Allen Brain Observatory. Reference for **Analysis 1 & 2** (`utils.py`):
- `binned_tuning_curve_()` L258–287 — bin speed over `[0,30]`, `np.digitize`, per-bin mean ± SEM, `np.random.permutation` shuffle baseline.
- `make_tuning_curves3()` L292–353 — `spearmanr(y, x)` (L345) + `levene(y, shuf_y)` (L346); significance `p < 0.05`.
- `get_split_data_decoder2()` L676–694 — running: `mean>3 & min>0.5`; still: `|mean|<0.5 & max<3` (matches [`Plan.md`](Plan.md)).
- Modulation index L580–581 — `2*(R_run−R_still)/(R_run+R_still)` (**2×** ours); significance via `ranksums`.
- Optional extras: tuning-shape fit `gaus_model_comparison()` L355–396; reliability `calculate_reliability()` L764–797; `+2`-frame onset lag L431.
- ⚠️ Its decoder is hard-wired to `DRIFTING_GRATINGS` and it *pools* natural + all gratings into one condition for tuning — do **not** copy that pooling.

**[jcbyts/V1Locomotion](https://github.com/jcbyts/V1Locomotion)** — Liska/Yates, MATLAB. Reference for **Analysis 3**:
- `Code/tent_basis.m` L1–19 — `B = max(1 − |ydiff/dscale|, 0)`. Port to Python.
- `Code/build_design_matrix.m` L66–231 — stimulus basis + drift (20 tents, L190) + running-onset regressors.
- `Code/AltLeastSqGainModel.m` L76–116 — alternating least squares; ReLU gain `g = max(g0 + g1*V, 0)` (L95–97).
- `Code/ridgeMML.m` — auto-λ ridge (or `sklearn.RidgeCV`).
- `Code/do_regression_ss.m` L120–198 — nested-model list + 5-fold CV + `rsquared()` = `1 − SS_res/SS_tot`.
- ⚠️ Stimulus model is 100% parametric gratings (direction/SF/phase/speed); data ingest is hard-filtered to `drifting_gratings`. **No natural-image code path exists** — only its stimulus-agnostic scaffolding (tent basis, nuisance regressors, gain-model machinery) transfers.

## Already done — do NOT rebuild

`load_data()` L16–47 · `extract_trials()` L152–240 (parameterized `response_window=(offset,duration)`, handles spontaneous; keeps stimuli separate) · `TrialData` L123–149 · `Plotter` L248–448 · `get_condition_intervals()` L70–115. All three analyses consume the same `TrialData`, so they are independent.

---

## Phase 0 — Shared prerequisites (do together first)

- [x] Obtain `visual_coding_data.npz`, place in `../data/` (beside the repo); confirm `load_data()` + `extract_trials()` run for all four stimuli in [`visual_coding.ipynb`](visual_coding.ipynb). **Blocks everyone.**
- [ ] Fix response windows once: `dg=(10,60)`, `sg/ns=(5,7)`, `spontaneous=(0,60)` (Plan.md §1). ⚠️ On the ~7-frame sg/ns trials, offset 5 + duration 7 runs *past* stimulus offset into the slow GCaMP6f decay and the next trial — choose this deliberately (the calcium transient outlasts the 0.23 s stimulus; per-image averaging over randomized repeats mitigates cross-trial bleed). See "Grating vs Natural" below.
- [ ] **Decide blank-sweep handling for `natural_scenes`:** the `frame` column codes the grey/blank sweep as `-1` (the 118 real images are `0–117`). `extract_trials` passes `frame` through untouched and nothing in the repo filters it — **exclude (or separately label) `frame == -1` trials** before any per-image f(S) or tuning.
- [ ] Add two shared reducers to `utils.py` (owner: **A**, reviewed by all):
  - `trial_mean_response(td) -> (n_cells, n_trials)` — mean ΔF/F over the response window.
  - `trial_mean_speed(td) -> (n_trials,)` — mean speed per trial (keep per-frame speed for run/still min/max tests).

---

## Person A — Analysis 1: `SpeedTuning` (medium) + shared reducers

Owns `utils.py` L456–551 + the Phase-0 reducers.
- [ ] **A1** `compute_tuning(n_bins=20)` (L484–498) → `bin_centers`, `mean_responses`, `std_responses`.
- [ ] **A2** `significance_test(n_shuffles=1000)` (L500–516) → `p_values`, `significant_mask` (shuffle speed labels, recompute tuning, **Levene** observed-vs-shuffled per cell, `p<0.05`).
- [ ] **A3** `compute_spearman()` (L518–526) → `rho`, `rho_p_values` (`scipy.stats.spearmanr`).
- [ ] **A4** `plot_tuning_curve(cell=None)` (L530–543) — single cell (mean±std) and population heatmap.
- [ ] **A5** `plot_significant_neurons()` (L545–551).
- [ ] **A6** Run A1–A3 **separately for each stimulus** (dg, sg, ns, spontaneous — never pooled); report fraction of significantly-tuned cells per stimulus. For ns, drop blank (`frame==-1`) trials first. With only ~47 cells (and ~34% of Allen neurons unresponsive to all stimuli; de Vries 2020), consider restricting to responsive cells via `p_dg/p_sg/p_ns` before interpreting fractions.
- [ ] **A7 (stretch)** Port Christensen's tuning-shape classifier (`gaus_model_comparison`).

**Done when:** per-stimulus significant-tuned fraction produced; sign of `rho` broadly agrees with `neurons_metadata.csv` responsiveness (`p_dg/p_sg/p_ns`).

---

## Person B — Analysis 2: `BinaryModulation` (light) + integration & validation

Owns `utils.py` L559–661 + cross-cutting deliverables.
- [ ] **B1** `classify_trials(run_threshold=3.0, still_threshold=0.5)` (L592–613) → run/still/ignored masks (running `mean>3 & all>0.5`; still `mean<0.5 & all<3`). Note: on ~7-frame sg/ns trials the "all-frames" guards are near-inert (speed ≈ constant within 0.23 s) — see "Grating vs Natural".
- [ ] **B2** `compute_mi()` (L615–628) → `mi = (R_run−R_still)/(R_run+R_still)` per cell (**our** formula, not 2×).
- [ ] **B3** `fit_gain_model()` (L630–640) → `gain_a`, `gain_b`. **Gotcha:** fit `R_run = a·R_still + b` **across stimulus conditions** (one (still, run) point per orientation / image / SF), not from a single scalar pair. For ns the conditioning variable is image identity (`frame`, blank excluded). Document the choice.
- [ ] **B4/B5** `plot_scatter(cell=None)` (L644–653) with fit line; `plot_mi_histogram()` (L655–661) with median.
- [ ] **B6** Cross-stimulus comparison (**the headline result**): compute MI for all four, then frame the key contrast as **gratings (dg + sg) vs natural (ns)**, with **spontaneous** as the no-stimulus baseline — this is the grating-vs-natural question the project exists to answer. Report dg and sg individually too. State the comparability caveats (below). Literature anchor: running ≈ **1.9× median evoked increase, ~13% of cells significantly modulated** (de Vries 2020, gratings); whether the natural-scenes MI distribution differs from gratings has **no published precedent** — it is the project's novel result ([`REFERENCES.md`](REFERENCES.md)).
- [ ] **B7** Validation: correlate `mi` vs `neurons_metadata.csv` `run_mod_dg/sg/ns` (+ p-values); flag divergences.
- [ ] **B8** Integration: assemble the final results section of `visual_coding.ipynb`; coordinate write-up.
- [ ] **B9 (assist C)** Own the shared **k-fold CV splitter** + box/violin plot helper (reused by Analysis 3).

**Done when:** the gratings-vs-natural MI comparison exists and `mi` is positively correlated with `run_mod_*`.

---

## Person C — Analysis 3: `EncodingModel` (heavy, reference-faithful, decomposed)

Owns `utils.py` L669–741. Port V1Locomotion's *machinery*; use B's CV splitter + plot helper.
- [ ] **C1** Tent basis — port `tent_basis.m` to `tent_basis(x, centers)`; unit-test vs hand values.
- [ ] **C2** Stimulus design `f(S)` — **gratings vs natural differ fundamentally, and neither reference repo helps here** (V1Locomotion is grating-only; Christensen fits no f(S)). Build from Plan.md:
  - dg: one-hot(direction × TF); sg: one-hot(orientation × SF × phase).
  - **ns (per [Plan.md](Plan.md) L32): define `f(S)` = per-image mean response** (118 images via `frame`, blank `-1` excluded) — subtract the per-image mean and fit the running terms to the **residual**. Mind the dimensionality: 118 conditions, ~50 trials/image.
  - spontaneous: constant.
- [ ] **C3** Drift baseline β₀(t) — `n_basis` tent functions over trial/session time (default 20).
- [ ] **C4** Running terms — additive `β_add·V`; multiplicative ReLU gain `max(1 + β_mult·V, 0)` on `f(S)`. Expect **both** terms: grating data show ~38% of cells multiplicative (gain ≈1.5) + ~27% additive (Dadarlat & Stryker 2017). For folding running into a natural-image model, see the V1T template (Li et al. 2023) in [`REFERENCES.md`](REFERENCES.md).
- [ ] **C5** Fitter — ridge auto-λ (`sklearn.RidgeCV` or port `ridgeMML`) + alternating least squares for the gain (~5 iters).
- [ ] **C6** `fit_all()` (L707–717) — four nested models (Null/Add/Mult/Full) per cell; **cross-validated R²** via B9 → `r2_null/add/mult/full`.
- [ ] **C7** `r2_decomposition()` (L719–731) → `delta_add/mult/full = model − null` (on CV R²).
- [ ] **C8** `plot_r2_decomposition()` (L735–741) — box/violin per model, **split by stimulus (gratings vs natural)**.

**Done when:** four models fit per cell with CV R²; ΔR² sane (`full ≥ add,mult`; not wildly negative); ns residual model runs with blank frames excluded.

---

## ⚠️ Grating vs Natural — stimulus-specific handling (the crux of the comparison)

The project's whole point is comparing running modulation between **gratings** (`drifting_gratings`, `static_gratings`) and **natural scenes** (`natural_scenes`), against a `spontaneous` baseline. Verifying both reference repos + our own data surfaced natural-scenes-specific issues that are easy to miss — read before starting.

- **No reference for the natural-scenes encoding model.** Both repos are grating-only: V1Locomotion's design matrix is hard-filtered to `drifting_gratings` (direction/SF/phase/speed) with zero natural-image code paths; Christensen fits no `f(S)` at all and actively *pools* natural + all gratings into one condition. So the natural-scenes `f(S)` (C2) is built from scratch per Plan.md:32 — per-image mean, predict the residual.
- **Literature predicts a *natural-scene-specific* running signature (and it's our novelty).** Froudarakis et al. 2014 (co-authored by our PI Berens) showed natural-scene population sparsening is **state-dependent** — present in the active/aroused state but not quiet wakefulness. Since running defines the active state, running may modulate natural-scene coding differently from gratings. No paper measures running modulation for natural scenes vs gratings on the same Allen cells → this comparison is the project's likely novel contribution ([`REFERENCES.md`](REFERENCES.md)).
- **Do NOT pool stimuli.** Our `extract_trials` keeps them separate (good). Don't replicate Christensen's merge — it would erase the very comparison we want.
- **Blank sweeps (`frame == -1`).** natural_scenes = 118 images (`0–117`) plus a grey blank `-1`; nothing in the repo filters it. Exclude before any per-image f(S)/tuning (Phase 0).
- **Trial length is not comparable across stimuli** (confirmed from the data): dg ≈ 60 frames (2 s); sg/ns ≈ 7 frames (0.23 s). Consequences:
  - *Run/still guards near-inert for sg/ns.* Within a 0.23 s trial the mouse's speed barely changes, so the "no frame < 0.5 / no frame > 3" tests collapse to thresholding the single trial-mean (unlike dg, where a 2 s trial can span both states). Classification for sg/ns is effectively coarser — note it, don't assume the guards did real work.
  - *Response window overruns short trials.* offset 5 on a ~7-frame trial samples into the GCaMP decay and the next trial; mitigated by averaging over randomized repeats, but decide the window deliberately (Phase 0).
  - *Speed-tuning behavioral timescale differs.* Each sg/ns tuning point is a near-instantaneous speed sample; each dg point is a sustained locomotor state.
- **Comparability caveat (state it in the write-up).** gratings vs natural differ simultaneously in trial length, behavioral timescale, stimulus dimensionality (~40 grating conditions vs 118 images; ~15 vs ~50 trials/condition), and f(S) form. When attributing a difference in running modulation to *stimulus type*, control for these (matched windows/normalization) and name the confounds.

---

## Gotchas (all)

- **Per-trial reduction is shared** (Phase 0) — reduce `(n_cells,n_trials,duration)` to a per-trial scalar the same way everywhere.
- **Grating vs natural needs special handling** — see the dedicated section above (no repo reference for ns `f(S)`; filter blank frames; short-trial caveats; comparability confounds).
- **Gain model (B3)** fit across conditions, not one scalar pair.
- **ΔR² uses cross-validated R²** — otherwise the Full model always wins by overfitting.
- **MI** is `(R_run−R_still)/(R_run+R_still)`; Christensen's decoding MI has an extra ×2.
- **Validation anchor**: `neurons_metadata.csv` `run_mod_*`, `osi_*`, `dsi_*`, `reliability_*` are pre-computed Allen metrics.
- **Statistical power / cell count.** ~47 matched cells, ~34% baseline-unresponsive, ~13% running-modulated (de Vries 2020) — expect effects in a minority; restrict to responsive cells (`p_*`, `reliability_*`) and don't over-interpret null results. See [`REFERENCES.md`](REFERENCES.md).

## Verification (end-to-end)

1. `conda activate allensdk`; run `visual_coding.ipynb` top-to-bottom with the `.npz` in `../data/`.
2. **A1**: per-stimulus significant-tuned fraction printed (each stimulus separate; ns blank excluded); tuning/significance plots render.
3. **A2**: `compute_mi` across all four stimuli; scatter slope `a`≈gain; MI median printed; `mi` correlates with `run_mod_*`.
4. **A3**: `fit_all` completes for all cells; finite ΔR² with `full ≥ add,mult`; ns residual model runs; decomposition plot renders.
5. **Research question**: one figure/table contrasts running modulation (MI + ΔR²_mult) for **gratings (dg, sg) vs natural (ns)**, with spontaneous as baseline — comparability caveats stated explicitly.

## Milestones

1. **M1** — Phase 0 done (data loads; windows + blank-frame policy + reducers + CV splitter merged). A1–A3, B1–B2, C1–C3 in progress on separate branches.
2. **M2** — all three analyses produce per-stimulus results; metadata validation passes (B7).
3. **M3** — gratings-vs-natural comparison (B6); plots polished (A4/A5, B4/B5, C8); write-up assembled (B8).
