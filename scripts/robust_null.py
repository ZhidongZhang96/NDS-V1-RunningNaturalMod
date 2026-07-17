"""Circular time-shift null (purged blocked CV) for dg and ns.

Shifting V by a random offset preserves running's SLOW AUTOCORRELATION but decouples it
from the response in time. If the ns ΔR² were merely slow calcium/arousal autocorrelation
leaking across densely-packed trials, the shifted-V surrogate would reproduce it (its slow
structure still leaks); if the observed ΔR² exceeds the shift null, the effect is genuinely
time-locked to running. This is the correct null for an autocorrelated regressor (the i.i.d.
V-permutation is not, because it destroys the autocorrelation the confound rides on).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
import utils
from utils import EncodingModel, TrialData

t0 = time.time()
def log(*a): print(*a, flush=True)
def wg(x):
    x = np.asarray(x)[np.isfinite(x)]
    try: return wilcoxon(x, alternative="greater").pvalue
    except ValueError: return np.nan

def blocked_splits(n, n_folds=5, gap=20):
    idx = np.arange(n); b = np.linspace(0, n, n_folds + 1).astype(int); out = []
    for k in range(n_folds):
        lo, hi = b[k], b[k + 1]
        out.append((np.concatenate([idx[:max(0, lo - gap)], idx[min(n, hi + gap):]]), idx[lo:hi]))
    return out

def shift_td(td, k):
    return TrialData(stimulus=td.stimulus, params=td.params, responses=td.responses,
                     running_speed=np.roll(td.running_speed, k, axis=0),
                     time=td.time, stimulus_params=td.stimulus_params)

def med_deltas(td, splits):
    em = EncodingModel(td, n_basis=5).fit_all(splits=splits)
    da, dm, df = em.r2_decomposition()
    return np.median(dm), np.median(df)

data = utils.load_data()
TD = {s: utils.extract_trials(data, s, response_window=None)
      for s in ("drifting_gratings", "natural_scenes")}
rng = np.random.default_rng(1)
out = {}
for s, B in (("drifting_gratings", 200), ("natural_scenes", 80)):
    td = TD[s]; n = td.responses.shape[1]; sp = blocked_splits(n, 5, 20)
    obs_m, obs_f = med_deltas(td, sp)
    lo = max(5, n // 10)
    mults, fulls = [], []
    for b in range(B):
        k = int(rng.integers(lo, n - lo))
        m, f = med_deltas(shift_td(td, k), sp)
        mults.append(m); fulls.append(f)
        if (b + 1) % 20 == 0: log(f"  {s} {b+1}/{B} [t+{time.time()-t0:.0f}s]")
    mults, fulls = np.array(mults), np.array(fulls)
    p_f = (np.sum(fulls >= obs_f) + 1) / (len(fulls) + 1)
    p_m = (np.sum(mults >= obs_m) + 1) / (len(mults) + 1)
    out[f"{s}_shift_full"] = fulls; out[f"{s}_shift_mult"] = mults
    out[f"{s}_obs_full"] = obs_f; out[f"{s}_obs_mult"] = obs_m
    log("== %-17s obs ΔR²_full=%+.4f (shift null mean %+.4f [%.4f,%.4f], p=%.3g)  "
        "obs ΔR²_mult=%+.4f (null mean %+.4f, p=%.3g)"
        % (s, obs_f, fulls.mean(), np.percentile(fulls, 2.5), np.percentile(fulls, 97.5), p_f,
           obs_m, mults.mean(), p_m))
np.savez("data/robust_null.npz", **out)
log(f"[t+{time.time()-t0:.0f}s] DONE -> data/robust_null.npz")
