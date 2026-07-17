"""Paper-metrics on the POOLED V1 cohort (3 containers, cached), vs VISpm.
Answers: do the looser population-mean metrics show a V1 area effect the strict ΔR² misses?
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
def wg(x, alt="greater"):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0): return np.nan
    try: return stats.wilcoxon(x, alternative=alt).pvalue
    except Exception: return np.nan
def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan

CIDS = [511507650, 511509529, 511510650]
RUN, STILL = 3.0, 0.5
STIM = ["drifting_gratings", "static_gratings", "natural_scenes"]

def build(cid):
    exps = boc.get_ophys_experiments(experiment_container_ids=[cid])
    EXP = {("A" if "A" in e["session_type"] else "B" if "B" in e["session_type"] else "C"): e["id"] for e in exps}
    S = {}
    for sk in ("A", "B"):
        ds = boc.get_ophys_experiment_data(EXP[sk])
        ts, dff = ds.get_dff_traces(); dxcm, _ = ds.get_running_speed()
        n = min(len(ts), dff.shape[1], len(dxcm))
        rs = np.zeros((2, n)); rs[0] = np.nan_to_num(dxcm[:n])
        csid = np.asarray(ds.get_cell_specimen_ids())
        ev = np.asarray(boc.get_ophys_experiment_events(ophys_experiment_id=EXP[sk]))[:, :n]
        stims = {"A": ["drifting_gratings"], "B": ["static_gratings", "natural_scenes"]}[sk]
        S[sk] = dict(session_type=sk, t=np.asarray(ts)[:n], dff=np.asarray(dff)[:, :n], events=ev,
                     running_speed=rs, csid=csid, stim_epoch_table=None,
                     stim_tables={s: ds.get_stimulus_table(s) for s in stims})
    matched = np.intersect1d(S["A"]["csid"], S["B"]["csid"])
    iA = {c: i for i, c in enumerate(S["A"]["csid"])}; iB = {c: i for i, c in enumerate(S["B"]["csid"])}
    return S, matched, np.array([iA[c] for c in matched]), np.array([iB[c] for c in matched])

def data_for(S, rA, rB, matched, sig):
    dA, dB = dict(S["A"]), dict(S["B"]); dA["dff"] = S["A"][sig][rA]; dB["dff"] = S["B"][sig][rB]
    return {"sessions": {"A": dA, "B": dB}, "matched_cell_ids": matched}

MET = {}
for cid in CIDS:
    S, matched, rA, rB = build(cid)
    log(f"container {cid}: n={len(matched)}")
    for sig in ("dff", "events"):
        data = data_for(S, rA, rB, matched, sig)
        for s in STIM:
            td = utils.extract_trials(data, s, response_window=None)
            R = td.responses.mean(2); V = td.running_speed.mean(1); labels = EncodingModel(td)._condition_labels()
            run, still = V > RUN, V < STILL
            mr, ms = R[:, run].mean(1), R[:, still].mean(1)
            MET.setdefault((sig, s, "meanD"), []).append(mr - ms)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ms > 0, mr / ms, np.nan)
            MET.setdefault((sig, s, "ratio"), []).append(ratio)
            conds = np.unique(labels); nc = R.shape[0]
            stc = np.full((nc, len(conds)), np.nan); rtc = np.full((nc, len(conds)), np.nan)
            for j, c in enumerate(conds):
                cs, cr = (labels == c) & still, (labels == c) & run
                if cs.sum() >= 2: stc[:, j] = R[:, cs].mean(1)
                if cr.sum() >= 2: rtc[:, j] = R[:, cr].mean(1)
            gains = np.full(nc, np.nan)
            for i in range(nc):
                m = np.isfinite(stc[i]) & np.isfinite(rtc[i])
                if m.sum() >= 4 and stc[i, m].std() > 1e-9:
                    gains[i] = stats.linregress(stc[i, m], rtc[i, m]).slope
            MET.setdefault((sig, s, "gain"), []).append(gains)

log("\n== POOLED V1 paper-metrics (running>3 vs still<0.5) ==")
VISPM = {  # for reference (from earlier runs)
    ("dff", "drifting_gratings"): "VISpm dff: meanΔ+0.0002(.18) ratio1.14 gain0.14",
    ("dff", "static_gratings"): "VISpm dff: meanΔ+0.0002(.33) ratio1.08 gain0.34",
    ("dff", "natural_scenes"): "VISpm dff: meanΔ+0.0019(.028) ratio2.03 gain0.59",
    ("events", "drifting_gratings"): "VISpm ev: meanΔ+0.0010(1.5e-6) ratio1.04 gain0.08",
    ("events", "static_gratings"): "VISpm ev: meanΔ+0.0005(.005) ratio1.04 gain0.09",
    ("events", "natural_scenes"): "VISpm ev: meanΔ+0.0010(5e-5) ratio1.10 gain0.36",
}
for sig in ("dff", "events"):
    log(f"\n--- {sig} ---")
    for s in STIM:
        md = np.concatenate(MET[(sig, s, "meanD")]); rt = np.concatenate(MET[(sig, s, "ratio")]); gn = np.concatenate(MET[(sig, s, "gain")])
        log(f"  V1 {s:18s} meanΔ={np.nanmedian(md):+.4f}[p={wg(md):.2g}]  ratio={gmean(rt):.2f}  gain={np.nanmedian(gn):.2f}  n={int(np.isfinite(md).sum())}")
        log(f"     ({VISPM[(sig, s)]})")
log("DONE")
