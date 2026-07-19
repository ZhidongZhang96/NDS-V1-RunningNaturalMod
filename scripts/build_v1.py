"""Pull a real V1 (VISp) container matched to our cohort's line/layer (Cux2-CreERT2, L2/3-4)
   and run the EXACT same encoding pipeline (blocked CV) on dF/F and on Allen L0 events.
   Direct test: is the near-null running modulation specific to VISpm, or does it hold in V1?
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

# 1. pick a VISp + Cux2-CreERT2 container (match our line/layer), not failed, with A+B+C
conts = [c for c in boc.get_experiment_containers(targeted_structures=["VISp"], cre_lines=["Cux2-CreERT2"])
         if not c.get("failed_experiment_container", False)]
for cont in sorted(conts, key=lambda c: c["id"]):
    cid = cont["id"]
    exps = boc.get_ophys_experiments(experiment_container_ids=[cid])
    EXP = {("A" if "A" in e["session_type"] else "B" if "B" in e["session_type"] else "C"): e["id"] for e in exps}
    if "A" in EXP and "B" in EXP:
        break
log(f"chosen V1 container {cid}: area={cont['targeted_structure']} depth={cont['imaging_depth']} "
    f"cre={cont['cre_line']}  exps={EXP}")

def build_session(sk, eid):
    ds = boc.get_ophys_experiment_data(eid)                    # downloads NWB
    ts, dff = ds.get_dff_traces()
    dxcm, dxtime = ds.get_running_speed()
    n = min(len(ts), dff.shape[1], len(dxcm))
    ts, dff, dxcm = np.asarray(ts)[:n], np.asarray(dff)[:, :n], np.asarray(dxcm)[:n]
    rs = np.zeros((2, n)); rs[0] = np.nan_to_num(dxcm)
    csid = np.asarray(ds.get_cell_specimen_ids())
    ev = np.asarray(boc.get_ophys_experiment_events(ophys_experiment_id=eid))[:, :n]  # order = csid
    stims = {"A": ["drifting_gratings"], "B": ["static_gratings", "natural_scenes"]}[sk]
    d = dict(session_type=sk, t=ts, dff=dff, events=ev, running_speed=rs, csid=csid,
             stim_epoch_table=ds.get_stimulus_epoch_table(),
             stim_tables={s: ds.get_stimulus_table(s) for s in stims})
    log(f"  session {sk} (exp {eid}): dff{dff.shape} events{ev.shape} ncells={len(csid)}")
    return d

log("downloading + building sessions A, B ...")
S = {"A": build_session("A", EXP["A"]), "B": build_session("B", EXP["B"])}
matched = np.intersect1d(S["A"]["csid"], S["B"]["csid"])
log(f"cells matched across A,B: n={len(matched)}")
idxA = {c: i for i, c in enumerate(S["A"]["csid"])}
idxB = {c: i for i, c in enumerate(S["B"]["csid"])}
rowA = np.array([idxA[c] for c in matched]); rowB = np.array([idxB[c] for c in matched])

def make_data(sig):
    dA, dB = dict(S["A"]), dict(S["B"])
    dA["dff"] = S["A"][sig][rowA]; dB["dff"] = S["B"][sig][rowB]
    return {"sessions": {"A": dA, "B": dB}, "matched_cell_ids": matched}

def wg(x):
    x = np.asarray(x)[np.isfinite(x)]
    try: return stats.wilcoxon(x, alternative="greater").pvalue
    except ValueError: return np.nan
def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan

STIM = ["drifting_gratings", "static_gratings", "natural_scenes", "spontaneous"]
for sig in ("dff", "events"):
    data = make_data(sig)
    log(f"\n== V1 container {cid} — EncodingModel on {sig.upper()} (blocked CV), n={len(matched)} cells ==")
    for s in STIM:
        td = utils.extract_trials(data, s, response_window=None)
        da, dm, df = EncodingModel(td, n_basis=5).fit_all(cv="blocked").r2_decomposition()
        extra = ""
        if sig == "events" and s in ("drifting_gratings", "natural_scenes"):
            R = td.responses.mean(2); V = td.running_speed.mean(1)
            run, still = V > 3, V < 0.5
            mr, ms = R[:, run].mean(1), R[:, still].mean(1); pos = ms > 0
            extra = f"   [rate-ratio={gmean(mr[pos]/ms[pos]):.2f}]"
        log(f"  {s:18s} add {np.median(da):+.4f}(p={wg(da):.2g})  mult {np.median(dm):+.4f}(p={wg(dm):.2g})  "
            f"full {np.median(df):+.4f}(p={wg(df):.2g}){extra}")
log("\nDONE")
