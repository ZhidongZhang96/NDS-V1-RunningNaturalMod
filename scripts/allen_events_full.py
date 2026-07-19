"""Canonical cross-check: re-run the encoding model + paper-metrics on Allen's official
   L0-deconvolved EVENTS (not our AR(1) proxy). Cells are mapped to event rows by signal
   correlation (events are deconvolved from the same dF/F), avoiding any NWB download.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
from scipy import stats
import utils
from utils import EncodingModel
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

def log(*a): print(*a, flush=True)
boc = BrainObservatoryCache(manifest_file="boc/manifest.json")

data = utils.load_data()
cell_ids = [int(c) for c in np.asarray(data["matched_cell_ids"])]

# --- area check for ALL 47 cells ---
exps = boc.get_ophys_experiments(cell_specimen_ids=cell_ids)
conts = sorted({e["experiment_container_id"] for e in exps})
areas = sorted({e.get("targeted_structure") for e in exps})
log(f"containers for our 47 cells: {conts}   targeted areas: {areas}")
EXP = {("A" if "A" in e["session_type"] else "B" if "B" in e["session_type"] else "C"): e["id"]
       for e in exps if e["experiment_container_id"] == conts[0]}
log(f"experiment ids: {EXP}")

def ar1(dff, g=0.95):
    s = dff.copy(); s[:, 1:] = dff[:, 1:] - g * dff[:, :-1]; s[:, 0] = 0.0
    return np.maximum(s, 0.0)

def zc(X):
    X = X - X.mean(1, keepdims=True); sd = X.std(1, keepdims=True)
    return X / np.where(sd == 0, 1, sd)

# --- map our cells to Allen event rows by correlation, per session, substitute dff ---
for sk in ("A", "B"):
    eid = EXP[sk]
    ev = boc.get_ophys_experiment_events(ophys_experiment_id=eid)   # (Nexp, T)
    dff = data["sessions"][sk]["dff"]                                # (47, T)
    log(f"session {sk}: events {ev.shape}  our dff {dff.shape}")
    if ev.shape[1] != dff.shape[1]:
        log(f"  !! timebase mismatch ({ev.shape[1]} vs {dff.shape[1]}) — aborting"); sys.exit(1)
    C = zc(ar1(dff)) @ zc(ev).T / dff.shape[1]                       # (47, Nexp) corr
    best = C.argmax(1); bestcorr = C.max(1)
    part = np.partition(C, -2, 1); margin = bestcorr - part[:, -2]
    log(f"  cell->event mapping: min best-corr={bestcorr.min():.2f}  min margin={margin.min():.2f}  "
        f"unique={len(set(best))==len(best)}")
    data["sessions"][sk]["dff"] = ev[best]                           # (47, T) Allen events, aligned

STIM = list(utils.STIMULI)
def wg(x):
    x = np.asarray(x)[np.isfinite(x)]
    try: return stats.wilcoxon(x, alternative="greater").pvalue
    except ValueError: return np.nan

log("\n== EncodingModel on ALLEN L0 EVENTS: ΔR² (blocked) + shuffled full (leak check) ==")
log(f"{'stimulus':18s}| blocked  add / mult / full                 | shuffled full")
for s in STIM:
    td = utils.extract_trials(data, s, response_window=None)
    da, dm, df = EncodingModel(td, n_basis=5).fit_all(cv="blocked").r2_decomposition()
    _, _, dfs = EncodingModel(td, n_basis=5).fit_all(cv="shuffled").r2_decomposition()
    log(f"{s:18s}| {np.median(da):+.4f}(p={wg(da):.2g}) {np.median(dm):+.4f}(p={wg(dm):.2g}) "
        f"{np.median(df):+.4f}(p={wg(df):.2g}) | {np.median(dfs):+.4f}(p={wg(dfs):.2g})")

# --- paper-metrics on Allen events ---
RUN_THR, STILL_THR = 3.0, 0.5
def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan
log("\n== PAPER-METRICS on ALLEN EVENTS (Run>3, Still<0.5) ==")
log(f"{'stimulus':18s}| mean Δ(run-still)[p]  geom-ratio(all)  autocorr dff->events")
for s in STIM:
    td = utils.extract_trials(data, s, response_window=None)
    R = td.responses.mean(2); V = td.running_speed.mean(1)
    run, still = V > RUN_THR, V < STILL_THR
    mr, ms = R[:, run].mean(1), R[:, still].mean(1)
    add = mr - ms; pos = ms > 0
    log(f"{s:18s}| {np.median(add):+.4f}[p={wg(add):.2g}]   {gmean(mr[pos]/ms[pos]):.2f}")
log("\nDONE")
