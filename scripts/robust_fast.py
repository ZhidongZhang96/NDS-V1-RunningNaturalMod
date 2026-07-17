"""Post-audit robustness + reconciliation (fast, point-estimate) tests.

Answers, with ground truth:
  (0) Per-session running distributions: is dg (Session A) actually under-run vs ns (B)?
  (1) Blocked / purged CV re-fit (dg,sg,ns): does the ns effect survive removing the
      calcium-autocorrelation leakage that shuffled KFold allows? Does ns>dg survive?
  (2) Classic population-mean running gain (running vs stationary trials, tuning removed):
      do dg cells show a positive mean gain (as the literature reports) even where the
      single-trial ΔR²_mult ~ 0 ?  -> the real reconciliation (metric strictness).
  (3) Synthetic-gain recovery on dg: inject a known ns-magnitude multiplicative gain into
      the real dg responses, re-fit at dg's N -> if recovered, dg *could* detect an ns-sized
      effect, so its real null is genuine (not merely under-powered).
  (4) A1/A2 subsample ns to dg's N (matched N / matched N+#cond) under blocked CV.
  (5) B1 cell-bootstrap 95% CI of the median ΔR².
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
import utils
from utils import EncodingModel, TrialData

t0 = time.time()
def log(*a): print(*a, flush=True);
def wg(x, alt="greater"):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0): return np.nan
    try: return wilcoxon(x, alternative=alt).pvalue
    except ValueError: return np.nan

def blocked_splits(n, n_folds=5, gap=0):
    """Contiguous test blocks (time order); training purged within `gap` trials."""
    idx = np.arange(n); b = np.linspace(0, n, n_folds + 1).astype(int); out = []
    for k in range(n_folds):
        lo, hi = b[k], b[k + 1]
        train = np.concatenate([idx[:max(0, lo - gap)], idx[min(n, hi + gap):]])
        out.append((train, idx[lo:hi]))
    return out

def fit_deltas(td, splits=None):
    em = EncodingModel(td, n_basis=5).fit_all(splits=splits)
    return em.r2_decomposition()

def subsample_td(td, idx):
    idx = np.sort(idx)  # keep temporal order for blocked CV
    sp = None if td.stimulus_params is None else {k: np.asarray(v)[idx] for k, v in td.stimulus_params.items()}
    return TrialData(stimulus=td.stimulus, params=td.params, responses=td.responses[:, idx, :],
                     running_speed=td.running_speed[idx], time=td.time[idx], stimulus_params=sp)

data = utils.load_data()
cell_ids = np.asarray(data["matched_cell_ids"])
TD = {s: utils.extract_trials(data, s, response_window=None) for s in
      ("drifting_gratings", "static_gratings", "natural_scenes")}
V = {s: td.running_speed.mean(axis=1) for s, td in TD.items()}
n_dg = TD["drifting_gratings"].responses.shape[1]
dg_cond = np.unique(np.column_stack([np.asarray(v, float) for v in
          TD["drifting_gratings"].stimulus_params.values()]), axis=0).shape[0]
log(f"[t+{time.time()-t0:.0f}s] loaded; n_dg={n_dg} dg_cond={dg_cond}")

# ---------- (0) per-session running distributions -----------------------------
log("\n== (0) PER-SESSION RUNNING (per-trial mean speed) ==")
for s in TD:
    v = V[s]
    log("  %-17s (Session %s)  n=%4d  mean=%5.2f  std=%5.2f  median=%5.2f  "
        "%%>1=%4.1f  %%>3=%4.1f  p95=%5.1f"
        % (s, "A" if s == "drifting_gratings" else "B", len(v), v.mean(), v.std(),
           np.median(v), 100*np.mean(v > 1), 100*np.mean(v > 3), np.percentile(v, 95)))

# ---------- (1) blocked / purged CV re-fit ------------------------------------
log("\n== (1) CV SCHEME COMPARISON (median ΔR²_full [p], ΔR²_mult [p], ΔR²_add [p]) ==")
RES = {}
for s in TD:
    n = TD[s].responses.shape[1]
    for tag, splits in (("shuffled", None),
                        ("blocked g0", blocked_splits(n, 5, 0)),
                        ("purged g20", blocked_splits(n, 5, 20))):
        da, dm, df = fit_deltas(TD[s], splits=splits)
        RES[(s, tag)] = (da, dm, df)
        log("  %-17s %-11s  full %+.4f [%.2g]   mult %+.4f [%.2g]   add %+.4f [%.2g]"
            % (s, tag, np.median(df), wg(df), np.median(dm), wg(dm), np.median(da), wg(da)))
    log("")

log("== ns>dg paired contrast under each CV scheme (one-sided Wilcoxon ns>dg) ==")
for tag in ("shuffled", "blocked g0", "purged g20"):
    for term, ti in (("mult", 1), ("full", 2)):
        dg = RES[("drifting_gratings", tag)][ti]; ns = RES[("natural_scenes", tag)][ti]
        try: p = wilcoxon(ns, dg, alternative="greater").pvalue
        except ValueError: p = np.nan
        log("  %-11s ΔR²_%-4s  ns(%+.4f) > dg(%+.4f)  p=%.3g" % (tag, term, np.median(ns), np.median(dg), p))

# ---------- (2) classic population-mean running gain --------------------------
log("\n== (2) POPULATION-MEAN RUNNING GAIN (running vs stationary, tuning removed) ==")
def running_gain(td, thr=1.0):
    R = td.responses.mean(axis=2)                       # (cells, trials)
    v = td.running_speed.mean(axis=1)
    em = EncodingModel(td); labels = em._condition_labels()
    resid = R.copy(); ratio_num = np.zeros(R.shape[0]); ratio_w = 0.0
    run = v > thr; still = ~run
    # tuning-removed residual difference (running - still), per cell
    for c in np.unique(labels):
        m = labels == c
        resid[:, m] = R[:, m] - R[:, m].mean(axis=1, keepdims=True)
    dmu = resid[:, run].mean(axis=1) - resid[:, still].mean(axis=1)   # (cells,)
    # per-condition ratio mean_run/mean_still averaged over conditions with both
    ratios = []
    for c in np.unique(labels):
        m = labels == c; r = m & run; s_ = m & still
        if r.sum() >= 2 and s_.sum() >= 2:
            mr, ms = R[:, r].mean(1), R[:, s_].mean(1)
            ok = np.abs(ms) > 1e-6
            ratios.append(np.where(ok, mr/ms, np.nan))
    gain = np.nanmedian(np.array(ratios), axis=0) if ratios else np.full(R.shape[0], np.nan)
    return dmu, gain, run.mean()
for s in TD:
    dmu, gain, frac = running_gain(TD[s])
    log("  %-17s  frac_running=%4.1f%%  Δμ(run-still) median=%+.4f  Wilcoxon(>0) p=%.3g  "
        "two-sided p=%.3g  gain-ratio median=%.3f"
        % (s, 100*frac, np.median(dmu), wg(dmu), wg(dmu, "two-sided"), np.nanmedian(gain)))

# ---------- (3) synthetic-gain recovery on dg ---------------------------------
log("\n== (3) SYNTHETIC ns-MAGNITUDE GAIN INJECTED INTO dg, re-fit at dg's N ==")
td_dg = TD["drifting_gratings"]
em0 = EncodingModel(td_dg); labels_dg = em0._condition_labels()
Rdg = td_dg.responses.mean(axis=2)
dhat = np.empty_like(Rdg)
for c in np.unique(labels_dg):
    m = labels_dg == c; dhat[:, m] = Rdg[:, m].mean(axis=1, keepdims=True)
dhat_c = dhat - dhat.mean(axis=1, keepdims=True)          # centered drive
vdg = td_dg.running_speed.mean(axis=1); vdg_c = vdg - vdg.mean()
win = td_dg.responses.shape[2]
log("  (target: reproduce ns ΔR²_mult≈+0.0019 / ΔR²_full≈+0.0033)")
for g in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
    inj = (g * vdg_c[None, :] * dhat_c)                  # (cells, trials) scalar add
    td_syn = TrialData(stimulus=td_dg.stimulus, params=td_dg.params,
                       responses=td_dg.responses + inj[:, :, None],
                       running_speed=td_dg.running_speed, time=td_dg.time,
                       stimulus_params=td_dg.stimulus_params)
    da, dm, df = fit_deltas(td_syn, splits=blocked_splits(n_dg, 5, 20))
    log("  g=%-4s  ΔR²_mult=%+.4f [p=%.2g]  ΔR²_full=%+.4f [p=%.2g]  frac full>0=%.0f%%"
        % (g, np.median(dm), wg(dm), np.median(df), wg(df), 100*np.mean(df > 0)))

# ---------- (4) A1/A2 subsample under blocked CV ------------------------------
log("\n== (4) SUBSAMPLE ns to dg's N (blocked/purged CV) ==")
rng = np.random.default_rng(0); td_ns = TD["natural_scenes"]; n_ns = td_ns.responses.shape[1]
uniq = np.unique(td_ns.stimulus_params["frame"])
def subsample_run(pick, B, tag):
    full, pf = [], []
    for b in range(B):
        sub = subsample_td(td_ns, pick())
        da, dm, df = fit_deltas(sub, splits=blocked_splits(sub.responses.shape[1], 5, 20))
        full.append(np.median(df)); pf.append(wg(df))
        if (b+1) % 50 == 0: log(f"   {tag} {b+1}/{B} [t+{time.time()-t0:.0f}s]")
    return np.array(full), np.array(pf)
A1f, A1p = subsample_run(lambda: rng.choice(n_ns, n_dg, replace=False), 150, "A1")
def pickA2():
    imgs = rng.choice(uniq, dg_cond, replace=False)
    pool = np.where(np.isin(td_ns.stimulus_params["frame"], imgs))[0]
    return rng.choice(pool, min(n_dg, len(pool)), replace=False)
A2f, A2p = subsample_run(pickA2, 150, "A2")
dg_full_obs = np.median(RES[("drifting_gratings", "purged g20")][2])
log("  A1 matched-N     : med-of-med ΔR²_full=%+.4f  frac p<0.05=%.0f%%  frac>dg_obs=%.0f%%"
    % (np.median(A1f), 100*np.mean(A1p < 0.05), 100*np.mean(A1f > dg_full_obs)))
log("  A2 matched-N+cond: med-of-med ΔR²_full=%+.4f  frac p<0.05=%.0f%%  frac>dg_obs=%.0f%%"
    % (np.median(A2f), 100*np.mean(A2p < 0.05), 100*np.mean(A2f > dg_full_obs)))

# ---------- (5) cell-bootstrap CI (purged-CV deltas) --------------------------
log("\n== (5) CELL-BOOTSTRAP 95%% CI of median ΔR²_full (purged CV) ==")
for s in TD:
    d = RES[(s, "purged g20")][2]
    bi = rng.integers(0, len(d), size=(20000, len(d)))
    meds = np.median(d[bi], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    log("  %-17s median %+.4f  95%% CI [%+.4f, %+.4f]  %s"
        % (s, np.median(d), lo, hi, "(incl 0)" if lo <= 0 <= hi else "(excl 0)"))

np.savez("data/robust_fast.npz",
         A1f=A1f, A1p=A1p, A2f=A2f, A2p=A2p, cell_ids=cell_ids,
         **{f"{s}__{tag}__{t}": RES[(s, tag)][i]
            for s in TD for tag in ("shuffled", "blocked g0", "purged g20")
            for i, t in enumerate(("add", "mult", "full"))})
log(f"\n[t+{time.time()-t0:.0f}s] DONE -> data/robust_fast.npz")
