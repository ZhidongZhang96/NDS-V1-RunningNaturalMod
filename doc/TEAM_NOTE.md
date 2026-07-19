# Team note — Analysis 3 (EncodingModel): CV fix, VISpm→V1 area finding, two-metric result

For Zhidong & Yuzhe. This summarises a substantial update to Analysis 3 (running-modulation
encoding model) on branch `encoding-model`. Two commits: **`6090b1d`** (correctness fix +
robustness) and **`d57977c`** (report repurposed around V1). Full detail in
[`EncodingModel.md`](EncodingModel.md).

## TL;DR

1. **The committed "natural-scene-specific multiplicative gain" was a cross-validation artifact** and is **retracted.** It came from a *shuffled* K-fold leaking slow calcium/running autocorrelation across densely-packed (0.27 s) trials. Fixed by defaulting `fit_all` to **leakage-free blocked CV**; the ns effect collapses from ΔR²_full +0.0033 (p=3e-5) to +0.0003 (n.s.).
2. **Our 47 cells are VISpm, not V1.** Container `511510753` is targeted at the posteromedial higher visual area, despite every doc saying "V1". (Layer/line are right — Cux2-CreERT2, 175 µm, L2/3–4 — only the area is off.)
3. **The real result needs two metrics.** The *population-mean running gain* (the papers' metric) is **strong, robust, and area-specific** in V1; the *cross-validated single-trial ΔR²* (our encoding metric) is **~null everywhere**. Both are true — a mean gain doesn't make individual held-out trials more predictable on noisy calcium.
4. **Report repurposed to V1** (pooled 3 VISp containers, n=363). VISpm is now the weaker-area comparison.

## 1. The CV-leakage fix (why the old result was wrong)

`fit_all` used `KFold(shuffle=True)`. On ns/static-gratings (Session B, trials ~0.27 s apart, within
GCaMP decay), each held-out trial is flanked in time by training trials, so slow autocorrelation leaks
and inflates ΔR². Drifting gratings (Session A, ~3 s apart) doesn't leak — which is exactly the pattern
we saw. **Fix:** `cv="blocked"` (contiguous, purged folds) is now the default; `cv="shuffled"`
reproduces the old numbers. Verified by: synthetic-gain recovery (blocked CV recovers a real injected
gain, so its nulls are genuine), a circular-shift null, and re-running on deconvolved spike events
(AR(1) + Allen L0) — the null holds on every signal. Under blocked CV **nothing survives FDR** in the
VISpm cohort.

## 2. The cohort is VISpm

`allensdk.ipynb` hard-codes a 47-cell list; all 47 resolve to container **511510753 = VISpm**. There's
no area filter anywhere in the repo — the cohort was chosen for cross-session matching, and it happens
to be a higher visual area. V1 (VISp) has 216 containers available; nothing forced VISpm. **Please
confirm whether VISpm was intended.** If V1 was the goal, the main data (`visual_coding_data.npz`) should
be rebuilt from a V1 container.

## 3. The two-metric result (the actual finding)

Pooled **V1** cohort: 3 VISp / Cux2 / 175 µm containers (`511507650`, `511509529`, `511510650`), n=363,
matched across sessions. Metrics on Allen L0 events (spike-comparable):

| metric | V1 dg | V1 sg | V1 ns | VISpm dg | VISpm sg | VISpm ns |
|---|---|---|---|---|---|---|
| running/stationary **rate ratio** | 1.57 | 1.72 | 2.15 | 1.04 | 1.04 | 1.10 |
| **ΔR²_full** (single-trial CV) | ~0/neg | ~0 | ~0 | ~0/neg | ~0 | ~0 |

- **Population gain:** V1 shows a robust ~1.5–2.5× gain (dg 1.57 ≈ Liska/Yates's 1.40), p=1e-20 to 1e-47
  across 363 cells / 3 containers — **and it's area-specific** (VISpm barely exceeds 1 for gratings,
  matching the de Vries 2020 "higher areas are less running-modulated outside L5" result).
- **Single-trial ΔR²:** null for **both** areas — only tiny additive sg/ns terms survive FDR
  (+0.0002–0.0003), validated by the Allen `run_mod` positive control (ns ρ=+0.30, **p=3e-5**). The
  multiplicative gain is not recoverable as cross-validated single-trial prediction, in V1 or VISpm.
- **Caution:** a single V1 container gave a large dg ΔR²_add (+0.0071) that **did not replicate** and
  vanished on pooling — always pool containers before trusting a per-container ΔR².

**Bottom line for the write-up:** running *does* modulate V1 (strong, area-specific, literature-matching
population gain); it simply isn't captured by the strict cross-validated single-trial ΔR² — a metric
limitation, not biological absence. Report both.

## 4. Decisions for the team

1. **Rebuild the project data on V1?** The report now leads with V1, but `visual_coding_data.npz`,
   `encoding_r2.npz`, and `visual_coding.ipynb` are still the VISpm 47-cell cohort. Decide: rebuild on V1,
   or keep VISpm data + the V1 report (flagged in §9).
2. **PR #1 conclusion changed** — these commits retract its "natural-scene-specific gain" headline.
   Review before merging.
3. **Framing:** `README`/`Plan.md`/`CLAUDE.md` still say "V1"; reconcile with the VISpm/area finding
   once (1) is decided.
4. **Session C demo notebook** (Zhidong's PR ask) still to do.

## Reproduce

Analysis scripts are in [`scripts/`](../scripts) (run from repo root, `allensdk` env). V1 pipeline:
`scripts/v1_report.py` (fits + pooling), `scripts/compute_v1_metrics.py` (population metrics),
`scripts/make_v1_gain_fig.py` (Figure 1). VISpm/robustness: `robust_fast.py`, `robust_null.py`,
`deconv_ar1.py`, `allen_events_full.py`, `build_v1.py`. Allen NWB/event downloads are fetched on demand
(gitignored).
