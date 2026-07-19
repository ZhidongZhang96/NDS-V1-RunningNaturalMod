"""Compute the ORIGINAL PAPERS' metrics on our 47 cells, for a like-for-like comparison.

Dadarlat & Stryker 2017 (ephys, 409 units):
  - mean response change running vs still (+62% firing rate);
  - additive/multiplicative decomposition: regress running tuning curve on still tuning curve
    per cell -> slope=gain (~1.5 in 38% of cells), intercept=offset;
  - population decoding of stimulus, running vs still (LDA, CV; 32-44% error drop).
Liska/Yates 2024 (Allen Neuropixels, ~1168 units):
  - geometric-mean running/still rate ratio (1.52 preferred / 1.40 all);
  - running vs PC1 correlation (median 0.407).

Our substrate is 2P calcium dF/F (can be negative), so ratios are computed only where the
denominator is positive; the robust bridge metrics are the tuning-curve regression gain and
the additive difference. Running = V>3 cm/s, stationary = V<0.5 cm/s (project thresholds).
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
from scipy import stats
sys.path.append(".")
import utils
from utils import EncodingModel

RUN_THR, STILL_THR = 3.0, 0.5
STIM = ["drifting_gratings", "static_gratings", "natural_scenes"]
data = utils.load_data()

def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan

print(f"Running = V>{RUN_THR} cm/s, stationary = V<{STILL_THR} cm/s\n")
rows = {}
for s in STIM:
    td = utils.extract_trials(data, s, response_window=None)
    R = td.responses.mean(axis=2)                      # (cells, trials) dF/F
    V = td.running_speed.mean(axis=1)
    labels = EncodingModel(td)._condition_labels()
    run, still = V > RUN_THR, V < STILL_THR
    ncell = R.shape[0]

    # ---- (1) mean response change (Dadarlat rate +62%) ----
    mr, ms = R[:, run].mean(1), R[:, still].mean(1)         # per cell
    add_diff = mr - ms                                      # additive (dF/F units)
    pos = ms > 0                                            # ratio only where still-mean>0
    pct = 100 * (mr[pos] - ms[pos]) / ms[pos]

    # ---- (2) multiplicative/additive decomposition (Dadarlat tuning regression) ----
    conds = np.unique(labels)
    still_tc = np.full((ncell, len(conds)), np.nan)
    run_tc = np.full((ncell, len(conds)), np.nan)
    for j, c in enumerate(conds):
        cs, cr = (labels == c) & still, (labels == c) & run
        if cs.sum() >= 2: still_tc[:, j] = R[:, cs].mean(1)
        if cr.sum() >= 2: run_tc[:, j] = R[:, cr].mean(1)
    gains, offs, sig_up = [], [], 0
    for i in range(ncell):
        m = np.isfinite(still_tc[i]) & np.isfinite(run_tc[i])
        if m.sum() >= 4 and np.std(still_tc[i, m]) > 1e-6:
            lr = stats.linregress(still_tc[i, m], run_tc[i, m])
            gains.append(lr.slope); offs.append(lr.intercept)
            if lr.stderr and lr.stderr > 0:
                t = (lr.slope - 1) / lr.stderr             # test slope > 1
                if stats.t.sf(t, m.sum() - 2) < 0.05: sig_up += 1
    gains, offs = np.array(gains), np.array(offs)

    # ---- (3) geometric-mean rate ratio (Liska/Yates 1.40-1.52) ----
    ratio_all = gmean(mr[pos] / ms[pos])
    pref = np.nanargmax(np.where(np.isfinite(still_tc), still_tc, -np.inf), axis=1)
    rp = np.array([run_tc[i, pref[i]] / still_tc[i, pref[i]]
                   if np.isfinite(still_tc[i, pref[i]]) and still_tc[i, pref[i]] > 0
                   and np.isfinite(run_tc[i, pref[i]]) else np.nan for i in range(ncell)])
    ratio_pref = gmean(rp)

    # ---- (4) running vs PC1 correlation (Liska/Yates median 0.407) ----
    Rc = R - R.mean(1, keepdims=True)
    U, sv, Vt = np.linalg.svd(Rc.T, full_matrices=False)   # trials x cells
    pc1 = U[:, 0] * sv[0]
    if np.corrcoef(pc1, R.mean(0))[0, 1] < 0: pc1 = -pc1   # align to mean activity
    r_pc1 = np.corrcoef(pc1, V)[0, 1]

    rows[s] = dict(n_run=int(run.sum()), n_still=int(still.sum()),
                   add_med=np.median(add_diff), add_p=stats.wilcoxon(add_diff, alternative="greater").pvalue,
                   pct_med=np.median(pct), n_pos=int(pos.sum()),
                   gain_med=np.median(gains), gain_gt1=100*np.mean(gains > 1), sig_up=100*sig_up/ncell,
                   off_med=np.median(offs), n_gain=len(gains),
                   ratio_all=ratio_all, ratio_pref=ratio_pref, r_pc1=r_pc1)
    print(f"[{s}] n_run={rows[s]['n_run']} n_still={rows[s]['n_still']} | tuning-fit cells={len(gains)}/{ncell}")

# ---- (5) population direction decoding, run vs still (Dadarlat 32-44% err drop) ----
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
td = utils.extract_trials(data, "drifting_gratings", response_window=None)
R = td.responses.mean(axis=2).T                            # trials x cells
V = td.running_speed.mean(axis=1)
ori = np.asarray(td.stimulus_params["orientation"], float)  # direction
dec = {}
for state, mask in (("still", V < STILL_THR), ("run", V > RUN_THR)):
    keep = mask & np.isfinite(ori)
    y = ori[keep]
    uy, cnt = np.unique(y, return_counts=True); ok = np.isin(y, uy[cnt >= 5])
    acc = cross_val_score(LinearDiscriminantAnalysis(), R[keep][ok], y[ok], cv=5).mean()
    dec[state] = (acc, int(ok.sum()), len(uy[cnt >= 5]))

print("\n" + "=" * 92)
print("PAPER METRICS ON OUR 47 CELLS  (dF/F, 2P) — vs the papers' reported values")
print("=" * 92)
print(f"\n{'metric':42s} {'dg':>10s} {'sg':>10s} {'ns':>10s}   paper (ephys)")
def line(name, key, fmt, paper):
    print(f"{name:42s} {fmt.format(rows['drifting_gratings'][key]):>10s} "
          f"{fmt.format(rows['static_gratings'][key]):>10s} {fmt.format(rows['natural_scenes'][key]):>10s}   {paper}")
line("mean run-still Δ (dF/F, median)", "add_med", "{:+.4f}", "Dadarlat rate +62%")
line("  Wilcoxon(>0) p", "add_p", "{:.3f}", "p=1e-47")
line("median per-cell % change (still>0)", "pct_med", "{:+.0f}%", "+62% (mean)")
line("multiplicative gain (median slope)", "gain_med", "{:.2f}", "Dadarlat ~1.5")
line("  % cells gain>1", "gain_gt1", "{:.0f}%", "—")
line("  % cells signif. gain>1", "sig_up", "{:.0f}%", "38% multiplicative")
line("additive offset (median intercept)", "off_med", "{:+.4f}", "Dadarlat offset ~0.8")
line("geom-mean rate ratio (all cond)", "ratio_all", "{:.2f}", "Liska/Yates 1.40")
line("geom-mean rate ratio (preferred)", "ratio_pref", "{:.2f}", "Liska/Yates 1.52")
line("running–PC1 correlation r", "r_pc1", "{:+.2f}", "Liska/Yates 0.41")
print(f"\ndg direction decoding (5-fold CV acc): still={dec['still'][0]:.3f}  run={dec['run'][0]:.3f}  "
      f"(chance≈{1/dec['still'][2]:.3f}; err drop {100*(1-(1-dec['run'][0])/(1-dec['still'][0])):+.0f}%)   Dadarlat -32..44%")
print("\nContrast: our leakage-free cross-validated ΔR²_full is ~0/negative for every stimulus (§7).")
print("DONE")
