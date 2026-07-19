"""Make the calcium signal spike-comparable via AR(1) non-negative deconvolution, then
   re-run the encoding model (blocked AND shuffled) + the paper-metrics on inferred events.

AR(1) inversion: s[t] = ReLU(c[t] - g*c[t-1]), g = exp(-1/(tau*fps)); GCaMP6f tau~0.5 s @30 Hz
=> g~0.95. This whitens the slow GCaMP autocorrelation (the leakage source) and yields a
non-negative rate-like signal (so the papers' ratio metrics become well-defined).
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
from scipy import stats
import utils
from utils import EncodingModel

GAMMA = 0.95
def deconv(dff, g=GAMMA):
    s = dff.copy()
    s[:, 1:] = dff[:, 1:] - g * dff[:, :-1]
    s[:, 0] = 0.0
    return np.maximum(s, 0.0)

def wg(x):
    x = np.asarray(x)[np.isfinite(x)]
    try: return stats.wilcoxon(x, alternative="greater").pvalue
    except ValueError: return np.nan

def lag1(R):                       # mean over cells of trial-to-trial (temporal) lag-1 autocorr
    a = []
    for c in range(R.shape[0]):
        x, y = R[c, :-1], R[c, 1:]
        if x.std() > 0 and y.std() > 0: a.append(np.corrcoef(x, y)[0, 1])
    return np.mean(a)

data = utils.load_data()
STIM = list(utils.STIMULI)

# --- keep ORIGINAL dff trial responses (for the autocorrelation-whitening demo) ---
orig_R = {s: utils.extract_trials(data, s, response_window=None).responses.mean(2) for s in ("drifting_gratings", "natural_scenes")}

# --- deconvolve each session's dff in place -> events ---
for k, sess in data["sessions"].items():
    if isinstance(sess.get("dff"), np.ndarray):
        sess["dff"] = deconv(sess["dff"])

print(f"AR(1) deconvolution, gamma={GAMMA}  ->  non-negative events\n")
print("== trial-to-trial lag-1 autocorrelation of the response (whitening check) ==")
for s in ("drifting_gratings", "natural_scenes"):
    ev_R = utils.extract_trials(data, s, response_window=None).responses.mean(2)
    print(f"  {s:18s}  dff={lag1(orig_R[s]):+.3f}  ->  events={lag1(ev_R):+.3f}")

print("\n== EncodingModel on EVENTS: ΔR² (blocked, leak-free) and shuffled (leak check) ==")
print(f"{'stimulus':18s}| blocked  add / mult / full                 | shuffled full")
FITEV = {}
for s in STIM:
    td = utils.extract_trials(data, s, response_window=None)
    da, dm, df = EncodingModel(td, n_basis=5).fit_all(cv="blocked").r2_decomposition()
    _, _, dfs = EncodingModel(td, n_basis=5).fit_all(cv="shuffled").r2_decomposition()
    FITEV[s] = (da, dm, df, dfs)
    print(f"{s:18s}| {np.median(da):+.4f}(p={wg(da):.2g}) {np.median(dm):+.4f}(p={wg(dm):.2g}) "
          f"{np.median(df):+.4f}(p={wg(df):.2g}) | {np.median(dfs):+.4f}(p={wg(dfs):.2g})")

# --- paper-metrics on EVENTS (now non-negative -> ratios well-defined) ---
RUN_THR, STILL_THR = 3.0, 0.5
def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan

print("\n== PAPER-METRICS on EVENTS (Running>3, Still<0.5 cm/s) ==")
print(f"{'stimulus':18s}| mean Δ(run-still) [p]   geom-ratio(all/pref)   gain(med slope)  %sig>1")
for s in STIM:
    td = utils.extract_trials(data, s, response_window=None)
    R = td.responses.mean(2); V = td.running_speed.mean(1)
    labels = EncodingModel(td)._condition_labels()
    run, still = V > RUN_THR, V < STILL_THR
    mr, ms = R[:, run].mean(1), R[:, still].mean(1)
    add = mr - ms
    conds = np.unique(labels); ncell = R.shape[0]
    stc = np.full((ncell, len(conds)), np.nan); rtc = np.full((ncell, len(conds)), np.nan)
    for j, c in enumerate(conds):
        cs, cr = (labels == c) & still, (labels == c) & run
        if cs.sum() >= 2: stc[:, j] = R[:, cs].mean(1)
        if cr.sum() >= 2: rtc[:, j] = R[:, cr].mean(1)
    gains, sig = [], 0
    for i in range(ncell):
        m = np.isfinite(stc[i]) & np.isfinite(rtc[i])
        if m.sum() >= 4 and stc[i, m].std() > 1e-9:
            lr = stats.linregress(stc[i, m], rtc[i, m]); gains.append(lr.slope)
            if lr.stderr and lr.stderr > 0 and stats.t.sf((lr.slope - 1) / lr.stderr, m.sum() - 2) < 0.05: sig += 1
    gains = np.array(gains)
    pos = ms > 0
    ratio_all = gmean(mr[pos] / ms[pos])
    pref = np.nanargmax(np.where(np.isfinite(stc), stc, -np.inf), 1)
    rp = np.array([rtc[i, pref[i]] / stc[i, pref[i]] if np.isfinite(stc[i, pref[i]]) and stc[i, pref[i]] > 0
                   and np.isfinite(rtc[i, pref[i]]) else np.nan for i in range(ncell)])
    print(f"{s:18s}| {np.median(add):+.4f} [p={wg(add):.2g}]   {ratio_all:4.2f} / {gmean(rp):4.2f}          "
          f"{np.median(gains):4.2f}            {100*sig/ncell:.0f}%")

np.savez("data/encoding_events_ar1.npz",
         **{f"{s}__{t}": FITEV[s][i] for s in STIM for i, t in enumerate(("add", "mult", "full", "shuf_full"))})
print("\nsaved data/encoding_events_ar1.npz")
print("DONE")
