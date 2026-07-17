# EncodingModel (Analysis 3) — Methodology, Predictions & Interpretation

Predictive nested-model analysis of running-speed modulation of V1 ΔF/F, implemented
in `EncodingModel` (`utils.py`). This document defines the model and estimator, states
the hypotheses it tests, gives the quantitative expectations from prior work, and fixes
the decision rules by which the results support or disprove the main hypothesis.
Companion docs: [`Plan.md`](Plan.md) (math), [`REFERENCES.md`](REFERENCES.md) (literature),
[`TASKS.md`](TASKS.md) (work plan).

> **Headline (corrected).** Under **leakage-free blocked cross-validation** (the default; §3),
> **no stimulus shows reliable single-trial running modulation** of V1 ΔF/F in this 47-cell cohort —
> nothing survives FDR correction (§7). An earlier version of this analysis used *shuffled* K-fold CV
> and reported a natural-scene-specific multiplicative gain; that effect is a **temporal-autocorrelation
> leakage artifact** and disappears under blocked CV (§7.1, §8). This is a *null result, not a power
> failure*: the model recovers a synthetically-injected gain, and drifting gratings actually contains
> **more** running than natural scenes. See §9 for how this reconciles with the population-mean gain
> literature (a different, less strict quantity).

## 1. Hypotheses

Main hypothesis (project): *"Layer 2/3 and 4 neurons in mouse V1 are positively modulated by
locomotion … specifically in drifting grating; verify whether those modulations are present …
and how different they are under naturalistic stimuli"* ([`Plan.md`](Plan.md):2). Operationalised
as three testable claims, per stimulus S ∈ {`drifting_gratings` (dg), `static_gratings` (sg),
`natural_scenes` (ns), `spontaneous` (spont)}:

| # | Hypothesis | Statistic | H₀ |
|---|---|---|---|
| **H1** (presence) | Running carries response information beyond stimulus tuning + slow drift | ΔR²_full > 0 across the population | median ΔR²_full = 0 |
| **H2** (structure) | Part of the modulation is a multiplicative gain on the stimulus drive | ΔR²_mult > 0; ΔR²_mult vs ΔR²_add | median ΔR²_mult = 0 |
| **H3** (stimulus dependence — the crux) | Modulation magnitude/structure differs **gratings (dg, sg) vs natural (ns)**; spont is the no-stimulus baseline | ΔR²_mult(gratings) vs ΔR²_mult(ns) | equal distributions |

Directional prior: the locomotion-gain literature is grating-based (H1/H2 expected for dg, sg),
whereas the natural-scene case has **no precedent**: a positive natural-scene signature would be
novel and is predicted by state-dependent sparsening (Froudarakis et al. 2014).

## 2. Model

Per neuron *i*, trial *t*, four **nested linear** models (design assembled by `_build_design`):

```
Null :  r_i(t) = A_i·s(t) + β₀ + Σ_j b_ij φ_j(t)
Add  :        + β_add · V(t)
Mult :        + β_mult · [ V(t) · d̂_i(S) ]
Full :        + β_add · V(t) + β_mult · [ V(t) · d̂_i(S) ]
```

- **f(S) = A·s(t)** — the stimulus tuning, a **fitted, ridge-penalized one-hot** design: `s(t)` is a one-hot vector over stimulus conditions and `A_i` are per-condition weights fit per neuron — exactly the tuning term of Liska/Yates (`do_regression_ss.m`). Conditions: dg = orientation×temporal-frequency; sg = orientation×spatial-frequency×phase; ns = image identity (`frame`, 118 images, blank `-1` excluded by `extract_trials`); spont = single condition. **Ridge shrinks the noisy per-condition estimates** — this matters: an unpenalized/OLS tuning (or a coefficient-1 offset) injects per-condition noise for few-trial conditions and degrades the held-out Null-model R².
- **Baseline** — a **fitted, unpenalized intercept** β₀ plus a **slow-drift** term `Σ_j b_j φ_j(t)`, where φ_j are `n_basis` (=5) partition-of-unity tent functions over trial time (`tent_basis`). No constant design column; the intercept is the baseline and the tent basis captures drift around it.
- **V(t)** — per-trial mean running speed (raw; `extract_trials` does not clamp, so small tracking-noise negatives occur, inconsequential for a linear regressor).
- **Multiplicative term** — running gated by the stimulus drive: `β_mult·(V·d̂(S))`, where `d̂(S)` is the per-condition drive (the per-fold OLS one-hot mean). This is the first-order linearization of Plan.md's rectified gain `ReLU[1+β_mult·V]`, keeping all four models linear (a clean cross-validated ΔR² decomposition — the project's target quantity, which neither reference computes). We do **not** fit the single-scalar gain that scales the *fitted* drive by alternating least squares: the exploration found that gain negligible for gratings and unnecessary to restore the control (§7). β_mult > 0 ⇒ running amplifies stimulus responses.

## 3. Estimation (`fit_all`)

- **Ridge regression** per neuron: features z-scored, penalty λ chosen by **generalized cross-validation (GCV)**, and the **intercept (baseline) left unpenalised** (as in Liska/Yates's `ridgeMML`). A closed-form SVD solve (`_ridge_cv_predict`); the stimulus-only Null/Add designs are identical across cells, so all 47 neurons are fit in a single multi-target solve (Mult/Full are per-cell — the interaction column is cell-specific). Ridge curbs the extra-parameter overfitting that would otherwise let Full win trivially.
- **Cross-validation: leakage-free blocked folds (default, `cv="blocked"`, `gap=5`).** Five *contiguous time-block* folds, with the training set **purged** of trials within `gap` of each test block (`_cv_splits`). Calcium (GCaMP decay ≈ 0.5 s) and running are slowly autocorrelated; a shuffled/random `KFold` interleaves each held-out trial with its immediate temporal neighbours in the training set, so for **densely-packed stimuli** (natural scenes / static gratings, trials ≈ 0.27 s apart) the running regressor can predict a *leaked* slow component and ΔR² is inflated. Blocked folds hold out whole spans of time, so only a genuinely *stationary* running→response relationship generalises. `cv="shuffled"` reproduces the earlier (leaky) split for comparison only — **this choice changes the conclusions** (§7.1, §8). The tuning `A·s(t)` is fit jointly each fold; the multiplicative gate `d̂(S)` is recomputed from *training* trials only (`_fold_stimulus_mean`).
- **Cross-validated R²** (pooled out-of-fold): `R² = 1 − Σ_t (y − ŷ_cv)² / Σ_t (y − ȳ)²`, per neuron. R² < 0 is admissible and meaningful (model predicts worse than the mean).
- **ΔR²_x = R²_x − R²_null** for x ∈ {add, mult, full} (`r2_decomposition`).

## 4. Statistical inference (population level)

- **Per term, per stimulus:** one-sided **Wilcoxon signed-rank** test of {ΔR²_x} across the 47 neurons vs 0 (H₀: median = 0). Report median ΔR²_x, the fraction of neurons > 0, and p.
- **H3 (gratings vs natural):** compare {ΔR²_mult} for gratings vs ns with a **Mann–Whitney U** test (or a paired signed-rank test across the same matched neurons). This is the pre-specified primary comparison.
- **Responsiveness restriction (sensitivity):** repeat on stimulus-responsive neurons only (`p_dg/p_sg/p_ns` < 0.05, and/or `reliability_*`), since ~34% of Allen neurons are unresponsive and effects are expected in a minority.
- Report all three ΔR² terms for every stimulus; do not over-interpret single-neuron values.

## 5. Expected results (priors from prior work — grating-only caveat applies)

- **Effect size is modest and sparse.** On the Allen data ~13% of neurons are significantly running-modulated (de Vries et al. 2020); for gratings, ~38% of cells show multiplicative and ~27% additive modulation with mean gain ≈1.5 (Dadarlat & Stryker 2017). Expect small, right-skewed CV ΔR² with a population median modestly > 0 for dg/sg.
- **Gain preserves tuning** (Niell & Stryker 2010): the f̂(S) coefficient should stay ≈1 and stable — running rescales rather than reshapes responses; consistent with a multiplicative contribution (H2).
- **Nested sanity:** median ΔR²_full ≥ ΔR²_add and ≥ ΔR²_mult (ridge keeps Full from overfitting); ΔR² should not be strongly negative.
- **Spontaneous:** a single condition ⇒ f̂(S) constant ⇒ `V·f̂(S) ∝ V`, so **Mult ≡ Add**; report only the additive effect for spont (baseline modulation).
- **Natural scenes:** no prior encoding model exists; the numbers above are **gratings/ephys and must not be transferred to ns**. A non-trivial ns signature is the novel outcome and is predicted (Froudarakis et al. 2014: natural-scene population coding sparsens specifically in the active/running state).

## 6. Interpretation — decision rules (support vs disprove the hypothesis)

| Observation | Conclusion |
|---|---|
| ΔR²_full > 0, significant, for a meaningful fraction of neurons | **Supports H1** — running is encoded beyond stimulus + drift. |
| ΔR²_full ≈ 0 or negative (n.s.) for a stimulus | **Fails to support H1** for that stimulus — running adds no out-of-sample predictive value. |
| ΔR²_mult > 0 and ≳ ΔR²_add (population) | **Supports H2** — modulation is gain-like (consistent with Niell/Dadarlat). |
| ΔR²_add > ΔR²_mult, ΔR²_mult ≈ 0 | Modulation is additive/offset-like — running shifts baseline, not gain. |
| ΔR²_mult(dg, sg) > ΔR²_mult(ns), significant | **Supports the "specifically in drifting grating" claim** — gain modulation is grating-specific. |
| ΔR²_mult(ns) ≈ or > gratings | **Novel result** — running modulates natural-scene coding comparably/distinctly; aligns with Froudarakis's state-dependence prediction and refines the grating-specific view. |
| spont ΔR²_add > 0 | Running modulates stimulus-independent baseline activity. |

The headline deliverable is the gratings-vs-natural contrast of ΔR²_mult (and ΔR²_full), with spontaneous as baseline — one figure/table, comparability caveats stated (§9).

**Realized outcome (§7):** under leakage-free CV the top rows — *fails to support H1/H2* — hold for **every** stimulus; the "natural-scene novel result" row seen under shuffled CV was a cross-validation artifact (§7.1).

## 7. Results (observed on the current data)

Fitted with `EncodingModel(td, n_basis=5).fit_all()` per stimulus (47 matched cells; ridge one-hot tuning; **leakage-free blocked CV**, the default — §3); arrays in `data/encoding_r2.npz`. Medians across cells; one-sided Wilcoxon signed-rank vs 0.

| stimulus | ΔR²_add | ΔR²_mult | ΔR²_full | cells ΔR²_full > 0 |
|---|---|---|---|---|
| drifting_gratings | −0.0005 (p=.89) | −0.0071 (p≈1) | −0.0086 (p≈1) | 23 % |
| static_gratings | −0.0001 (p=.61) | −0.0007 (p=.98) | −0.0007 (p=.98) | 34 % |
| natural_scenes | +0.0006 (p=.038) | −0.0000 (p=.38) | +0.0003 (p=.25) | 51 % |
| spontaneous | +0.0000 (p=.44) | +0.0000 (p=.44) | +0.0001 (p=.38) | 53 % |

Under **Benjamini–Hochberg FDR (q=0.05)** across the 12 term×stimulus tests, **none survives** (smallest raw p = natural-scene additive, .038, vs a BH threshold of .004). Every ΔR²_full is ≤ 0 or non-significant; the grating terms are significantly *negative* — adding running columns *worsens* held-out prediction, i.e. the extra parameters do not pay for themselves.

![Cross-validated ΔR² decomposition by stimulus](figures/dR2_decomposition.png)

**Figure 1. Cross-validated ΔR² decomposition by stimulus (leakage-free blocked CV).** Median ΔR² (±95% bootstrap CI, n=47 cells) for the additive, multiplicative, and full running terms. No marker is filled (none reaches one-sided Wilcoxon *p*<0.05); every term scatters around or below zero.

- **H1 (presence) — not supported for any stimulus.** No ΔR²_full is significantly positive; both grating types are significantly negative.
- **H2 (gain structure) — not supported.** No multiplicative term is positive for any stimulus.
- **H3 (gratings vs natural) — no positive natural-scene advantage.** Natural-scene ΔR²_full (+0.0003) exceeds drifting gratings (−0.0086) at p=.017, but that reflects drifting gratings' *negative overfit*, not a positive natural-scene effect; natural scenes do **not** exceed static gratings (p=.18) and are not themselves significant.

### 7.1 The earlier natural-scene effect was a cross-validation leakage artifact

An earlier version of this analysis used **shuffled** K-fold CV and concluded that running modulation was *natural-scene-specific with a multiplicative gain*. That result does not survive leakage-free CV:

| stimulus | ΔR²_full **shuffled** (leaky) | ΔR²_full **blocked** (leak-free) |
|---|---|---|
| drifting_gratings | −0.0017 (p=.76) | −0.0086 (p≈1) |
| static_gratings | +0.0003 (p=.046) | −0.0007 (p=.98) |
| natural_scenes | **+0.0033 (p=3e-5)** | **+0.0003 (p=.25)** |

The natural-scene ΔR²_full collapses from +0.0033 (p=3e-5) to ≈0, its multiplicative term from +0.0019 (p=7e-4) to −0.0000, and the shuffled natural-scenes > gratings contrast (p=.002) becomes non-significant. Crucially, the collapse is confined to the **densely-packed Session-B stimuli** (natural scenes & static gratings, trials ≈ 0.27 s apart, within the GCaMP decay window) and spares the sparsely-packed drifting gratings (≈ 3 s apart) — the exact signature of temporal-autocorrelation leaking through random folds (§8, Fig. 3B).

**Conclusion.** With a reference-grade tuning model **and** leakage-free cross-validation, running does **not** add reliable out-of-sample single-trial predictive power for V1 ΔF/F for *any* stimulus type in this 47-cell cohort. The natural-scene modulation reported by the earlier (shuffled-CV) version was a cross-validation artifact. This is a *genuine null, not a power failure* (§8): the model recovers a synthetically-injected gain at drifting gratings' trial count, and the drifting-gratings session contains **more** running than the natural-scenes session. §9 explains why this does not contradict the population-mean gain literature.

## 8. Validation & controls

The central validation is that the result is **robust to cross-validation scheme** and that the flat ΔR² is a *genuine null*, not lost power. Figure 3 collects the checks.

![CV-leakage validation](figures/cv_leakage.png)

**Figure 3. Why the natural-scene effect was an artifact — and why the null is genuine.**
**(A)** Per-trial running distributions: Session A (drifting gratings) has *more* running than Session B (natural scenes) — mean 10.4 vs 5.4 cm/s; 44 % vs 23 % of trials > 3 cm/s — so the drifting-gratings null is not from lack of running.
**(B)** Median ΔR²_full under shuffled vs blocked CV: the natural-scene and static-gratings effects vanish once autocorrelation leakage is removed.
**(C)** Synthetic-gain recovery: injecting a known multiplicative running gain into the *real* drifting-gratings responses and re-fitting under blocked CV recovers it (g=0.05 → ΔR²_full +0.007, already exceeding the old shuffled natural-scene value).
**(D)** Circular-shift null: the observed blocked-CV ΔR²_full (dashed) sits *marginally above* its autocorrelation-preserving shift-null for both stimuli (dg p=.045; ns p=.012, above all 80 surrogates) yet remains ≤ 0 — a sub-threshold time-locked signal, not a positive effect.

- **CV-scheme robustness (primary).** Shuffled K-fold leaks calcium/running autocorrelation across the train/test boundary and inflates ΔR² for sub-second trial spacing; blocked+purged folds remove it (§3). All headline numbers use blocked CV.
- **Synthetic-gain recovery (positive/power control).** A genuine *stationary* gain injected into the real drifting-gratings data **is** recovered under blocked CV, so the blocked-CV nulls are not an artefact of over-conservative folds. Recovered ΔR²_full: g=0.02 → −0.001, g=0.05 → +0.007, g=0.1 → +0.021 (Fig. 3C).
- **Circular-shift null (autocorrelation-preserving).** Shifting V by a random offset keeps running's autocorrelation but decouples it from the response in time — the correct null for an autocorrelated regressor. The observed blocked-CV ΔR² sits *marginally above* this null for both drifting gratings (full p=.045, mult p=.020) and natural scenes (full p=.012 — above all 80 surrogates — mult p=.025): a **weak, genuinely time-locked** running signal exists beyond mere autocorrelation. But the observed ΔR² is itself ≤ 0 (running does not beat the tuning-only model), so this is a *sub-threshold whisper* — converging with the small additive natural-scene term on deconvolved events (above) — not a positive modulation.
- **Signal robustness — deconvolution (calcium → spikes).** The model runs on ΔF/F, whereas the reference papers use spikes. Re-running on inferred spike events — both a quick AR(1) non-negative deconvolution and **Allen's canonical L0 events** (what de Vries 2020 and the `run_mod_*` indices are built from) — leaves the blocked-CV ΔR²_full **null for every stimulus** (natural scenes closest at p≈.08 on L0 events; at most a weak *additive* natural-scene term survives across signals, L0 p≈.01). Deconvolution collapses the natural-scene trial-to-trial autocorrelation (**0.54 → 0.10**), confirming it as the leakage source — yet shuffled CV *still* inflates natural scenes on the sharp signal (L0: shuffled +0.0029 vs blocked +0.0009), so blocked CV is required regardless of signal. (Cells mapped to L0 event rows by signal correlation; a crude AR(1) deconvolution produced a spurious multiplicative term that did **not** replicate on the canonical events — hence two methods.)
- **Area control — matched V1 cohort.** Re-running the *identical* pipeline on a matched **V1** container (VISp, Cux2-CreERT2, 175 µm — same line/layer as our cohort; 91 cells matched across sessions by SDK `cell_specimen_id`) reveals a **significant additive running effect that our VISpm cohort lacks**: ΔR²_add = **+0.0071 for drifting gratings (p=1.6e-6**, leakage-free; +0.0013 on L0 events, p=2e-4) and +0.0013 for static gratings (p=1e-6), vs ≈0/negative in VISpm. The population rate ratio is also larger (V1 dg 1.85 vs VISpm 1.37). **Crucially, the multiplicative and full terms stay null in V1 too** (dg ΔR²_full p=.34; ΔR²_mult negative). So single-trial running modulation of upper-layer excitatory cortex is **area-dependent and *additive*** (present in V1, weak/absent in VISpm), whereas the hypothesised **multiplicative gain is undetectable by cross-validated single-trial ΔR² regardless of area** — the metric-strictness limit (§9), confirmed in bona fide V1. *(One container; a few more would confirm robustness.)*
- **Readout sensitivity.** V1's effect appears in **ΔR²_add but not ΔR²_full** — the noisy multiplicative column drags the full model down (they are redundant/collinear). **ΔR²_add is therefore the more sensitive readout** of running modulation than the pre-registered ΔR²_full; report both.
- **External control (now weak / underpowered).** Per-cell ΔR²_full vs the Allen `run_mod_*` index is small and non-significant for every stimulus (dg ρ=+0.09, sg +0.07, ns +0.14; all p>.35, n=47) — consistent with no reliable per-cell modulation, though underpowered at n=47.
- **Negative control.** Shuffling V across trials collapses ΔR²_add/mult to ≈ 0 (unchanged).

![Encoding-model ΔR²_full vs Allen run_mod index](figures/validation_runmod.png)

**Figure 2. External control (underpowered).** Per-cell ΔR²_full against the pre-computed Allen running-modulation index `run_mod_*`, per stimulus; red line = least-squares fit. Spearman ρ is small and non-significant for every stimulus (dg +0.09, sg +0.07, ns +0.14; all p>.35 at n=47) — consistent with no reliable per-cell modulation.

## 9. Limitations, confounds & reconciliation with prior work

- **⚠️ Area: the matched cells are VISpm, not V1.** All 47 cells belong to Allen experiment container **511510753, targeted area `VISpm`** (posteromedial higher visual area) — *not* primary visual cortex (VISp/V1), despite this project's "V1" framing. Running modulation is area-dependent (de Vries et al. 2020), and the reference papers (Niell & Stryker; Dadarlat & Stryker; Liska/Yates mouse data) are all V1 — so a weaker/different effect here is partly expected on area grounds alone, independent of the metric and calcium-vs-spike issues below. **A matched V1 control (§8, "Area control") makes this concrete: a V1 cohort matched on line/layer shows a significant *additive* running effect (drifting gratings ΔR²_add p≈1e-6) that this VISpm cohort lacks** — so the area genuinely matters and **rebuilding the dataset on a V1 (VISp) container is recommended** if V1 is the intended target. (The multiplicative gain stays null even in V1, so that part is the metric, not the area.) **This should be confirmed as intended** (it may be a cell-matching/data-extraction choice); if unintended it affects the whole project framing and every doc that says "V1".
- **Reconciliation with the locomotion-gain literature (why a null here is *not* a contradiction).** Classic reports (Niell & Stryker 2010; Dadarlat & Stryker 2017; de Vries et al. 2020) quantify a **population-mean gain** — mean response on running vs stationary trials, averaged over *thousands* of neurons. Our ΔR² is a far stricter quantity: the *cross-validated, single-trial* predictive value of running *beyond* stimulus tuning, per cell, on 47 calcium-imaged neurons. A real but modest mean gain can contribute ≈ 0 to single-trial ΔR² when trial-to-trial ΔF/F is noisy. Computing the papers' *own* metrics on our cells confirms this directly: on **Allen's canonical L0 events** the geometric-mean running/stationary rate ratio is **1.37 (drifting gratings) / 1.71 (natural scenes)** — matching Liska/Yates's 1.40 — and the mean run−stationary response is significantly positive (dg p=.006, ns p=.008), *while the cross-validated single-trial ΔR² on the same cells and same signal stays null*. Same cells, same signal, two metrics: the population rate-ratio is positive and paper-matching; the strict single-trial metric is null. So the null does **not** contradict the field — it reflects metric strictness (plus the area, cohort, and modality caveats in this section), not evidence that running fails to modulate visual cortex.
- **~47-neuron cohort, two-photon under-reporting.** Low power for a sparse, small effect; prefer population statistics and treat single-cell values cautiously.
- **Session / stimulus confounds for any cross-stimulus comparison** ([`TASKS.md`](TASKS.md):99–111): drifting gratings = Session A; static gratings & natural scenes = Session B — different recording day, running prevalence and arousal. Trial windows differ (dg ≈ 60 frames / 2 s vs sg,ns ≈ 7 frames / 0.23 s), so per-trial response and running SNR differ and the running regressor is a 2 s vs 0.23 s average. These confound any drifting-gratings-vs-Session-B contrast independently of the (now-removed) CV artifact; the *within-session* natural-scenes-vs-static-gratings comparison is cleaner and it, too, shows no effect.
- **Blocked-CV cost.** Contiguous folds reduce effective training data near block boundaries (purge) and test on temporally-clustered trials; the synthetic-recovery control (§8) shows this does *not* spuriously null a genuine stationary effect.
- **Linearised (not rectified) gain** — the interaction approximates `ReLU[1+β_mult V]`; the rectification and any strongly negative-V regime are not modelled (V is near-non-negative in practice).
- **Arousal vs locomotion** are dissociable (Vinck et al. 2015); running here is a proxy for the active state, not isolated motor drive.

## 10. References

Model & machinery: Liska/Yates (V1Locomotion, eLife 87736). Additive/multiplicative decomposition: Dadarlat & Stryker 2017. Gain preserving tuning: Niell & Stryker 2010. Allen dataset / running prevalence: de Vries et al. 2020. State-dependent natural-scene coding: Froudarakis et al. 2014. Running-as-input natural-image encoding template: Li et al. 2023 (V1T). Full citations and URLs in [`REFERENCES.md`](REFERENCES.md).
