# EncodingModel (Analysis 3) — Methodology, Predictions & Interpretation

Predictive nested-model analysis of running-speed modulation of V1 ΔF/F, implemented
in `EncodingModel` (`utils.py`). This document defines the model and estimator, states
the hypotheses it tests, gives the quantitative expectations from prior work, and fixes
the decision rules by which the results support or disprove the main hypothesis.
Companion docs: [`Plan.md`](Plan.md) (math), [`REFERENCES.md`](REFERENCES.md) (literature),
[`TASKS.md`](TASKS.md) (work plan).

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
whereas the natural-scene case has **no precedent** — a positive natural-scene signature would be
novel and is predicted by state-dependent sparsening (Froudarakis et al. 2014).

## 2. Model

Per neuron *i*, trial *t*, four **nested linear** models (design assembled by `_build_design`):

```
Null :  r_i(t) = f̂_i(S) + Σ_j b_ij φ_j(t)
Add  :        + β_add · V(t)
Mult :        + β_mult · [ V(t) · f̂_i(S) ]
Full :        + β_add · V(t) + β_mult · [ V(t) · f̂_i(S) ]
```

- **f̂(S)** — per-condition mean response (the empirical tuning). Conditions: dg = orientation×temporal-frequency; sg = orientation×spatial-frequency×phase; ns = image identity (`frame`, 118 images, blank `-1` already excluded by `extract_trials`); spont = single condition (constant). The Null model thus **"takes out" the stimulus mean and predicts the residual** ([`Plan.md`](Plan.md):32).
- **β₀(t) = Σ_j b_j φ_j(t)** — slow-drift baseline; φ_j are `n_basis` (=5) partition-of-unity tent functions over trial time (`tent_basis`). No separate intercept: the tent basis already spans the constant.
- **V(t)** — per-trial mean running speed (raw; `extract_trials` does not clamp, so small tracking-noise negatives occur — inconsequential for a linear regressor).
- **Multiplicative term.** Plan.md specifies a rectified gain `ReLU[1 + β_mult·V]` scaling f(S) (after Liska/Yates). We use its **first-order linearization**, the interaction `β_mult·(V·f̂(S))`, because (i) it keeps all four models linear, giving a clean cross-validated ΔR² decomposition — the project's target quantity, which neither reference paper computes — and (ii) `f̂(S)(1+β_mult V) = f̂(S) + β_mult(V·f̂(S))`. Sign and magnitude of β_mult retain the gain interpretation (>0 ⇒ running amplifies stimulus responses).

## 3. Estimation (`fit_all`)

- **Ridge regression** per neuron: `Pipeline(StandardScaler, RidgeCV(alphas=logspace(-3,3,13)))`; features standardised, penalty λ selected by RidgeCV. Ridge is required because the Full model's terms are partly collinear and because it curbs the extra-parameter overfitting that would otherwise let Full win trivially.
- **5-fold cross-validation** (`KFold`, shuffled, seed 0). Within each fold, **f̂(S) is recomputed from training trials only** (`_fold_stimulus_mean`); this is essential — using the all-trial mean would let the Null model memorise per-condition means and inflate every R².
- **Cross-validated R²** (pooled out-of-fold): `R² = 1 − Σ_t (y − ŷ_cv)² / Σ_t (y − ȳ)²`, per neuron. R² < 0 is admissible and meaningful (model predicts worse than the mean).
- **ΔR²_x = R²_x − R²_null** for x ∈ {add, mult, full} (`r2_decomposition`, to be added).

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

The headline deliverable is the gratings-vs-natural contrast of ΔR²_mult (and ΔR²_full), with spontaneous as baseline — one figure/table, comparability caveats stated (§8).

## 7. Validation & controls

- **Positive control (external):** per-neuron modulation (sign/magnitude of ΔR²_full or β_add) should correlate with the pre-computed Allen indices `run_mod_dg/sg/ns` (and agree with `p_run_mod_*`) in `data/neurons_metadata.csv`. Large divergence flags a methodological error.
- **Positive control (internal):** the fitted f̂(S) coefficient ≈ 1.
- **Negative control:** shuffling V across trials must collapse ΔR²_add and ΔR²_mult to ≈ 0 (running carries no information under the null).

## 8. Limitations & confounds

- **Gratings-vs-natural comparability** ([`TASKS.md`](TASKS.md):99–111): dg trials ≈ 60 frames (2 s) vs sg/ns ≈ 7 frames (0.23 s); ~40 grating conditions vs 118 images (~15 vs ~50 trials/condition); differing behavioural timescale. Differences in ΔR² across stimulus type are confounded by these; control with matched windows/normalisation and state the confound.
- **Linearised (not rectified) gain** — the interaction approximates `ReLU[1+β_mult V]`; the rectification and any strongly negative-V regime are not modelled (V is near-non-negative in practice).
- **Arousal vs locomotion** are dissociable (Vinck et al. 2015); running here is a proxy for the active state, not isolated motor drive.
- **Two-photon under-reporting** and **~47-neuron** sample limit power; treat null results cautiously and prefer population statistics.

## 9. References

Model & machinery: Liska/Yates (V1Locomotion, eLife 87736). Additive/multiplicative decomposition: Dadarlat & Stryker 2017. Gain preserving tuning: Niell & Stryker 2010. Allen dataset / running prevalence: de Vries et al. 2020. State-dependent natural-scene coding: Froudarakis et al. 2014. Running-as-input natural-image encoding template: Li et al. 2023 (V1T). Full citations and URLs in [`REFERENCES.md`](REFERENCES.md).
