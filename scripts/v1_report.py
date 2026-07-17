"""Full analysis battery on a V1 (VISp) cohort, to make V1 the report's main focus.
Primary cohort = one matched container (full battery); + 2 more V1 containers pooled for
the headline additive effect. Saves per-cell arrays -> data/encoding_v1.npz.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats
import utils
from utils import EncodingModel, TrialData
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

t0 = time.time()
def log(*a): print(*a, flush=True)
boc = BrainObservatoryCache(manifest_file="boc/manifest.json")
def wg(x, alt="greater"):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0): return np.nan
    try: return stats.wilcoxon(x, alternative=alt).pvalue
    except ValueError: return np.nan
def gmean(x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    return np.exp(np.mean(np.log(x))) if len(x) else np.nan
def blocked_gap(n, g=20):
    idx = np.arange(n); b = np.linspace(0, n, 6).astype(int); out = []
    for k in range(5):
        lo, hi = b[k], b[k + 1]
        out.append((np.concatenate([idx[:max(0, lo - g)], idx[min(n, hi + g):]]), idx[lo:hi]))
    return out

conts = [c for c in boc.get_experiment_containers(targeted_structures=["VISp"],
         cre_lines=["Cux2-CreERT2"], imaging_depths=[175]) if not c.get("failed_experiment_container", False)]
conts = sorted(conts, key=lambda c: c["id"])
PRIM = 511507650
CIDS = [PRIM] + [c["id"] for c in conts if c["id"] != PRIM][:2]
log(f"V1 VISp/Cux2/175um containers available: {len(conts)}; using {CIDS}")

def build(cid, with_spont=True):
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
        try:
            epoch = ds.get_stimulus_epoch_table() if with_spont else None
        except Exception:
            epoch = None
        S[sk] = dict(session_type=sk, t=np.asarray(ts)[:n], dff=np.asarray(dff)[:, :n], events=ev,
                     running_speed=rs, csid=csid, stim_epoch_table=epoch,
                     stim_tables={s: ds.get_stimulus_table(s) for s in stims})
    matched = np.intersect1d(S["A"]["csid"], S["B"]["csid"])
    iA = {c: i for i, c in enumerate(S["A"]["csid"])}; iB = {c: i for i, c in enumerate(S["B"]["csid"])}
    return S, matched, np.array([iA[c] for c in matched]), np.array([iB[c] for c in matched])

def data_for(S, rA, rB, matched, sig):
    dA, dB = dict(S["A"]), dict(S["B"])
    dA["dff"] = S["A"][sig][rA]; dB["dff"] = S["B"][sig][rB]
    return {"sessions": {"A": dA, "B": dB}, "matched_cell_ids": matched}

STIM = ["drifting_gratings", "static_gratings", "natural_scenes", "spontaneous"]
POOL = {}                       # (sig,stim,term) -> list of per-cell arrays (blocked, across containers)
prim = {}                       # primary-only extras
pooled_ids = []
for ci, cid in enumerate(CIDS):
    log(f"\n--- container {cid} ---  [t+{time.time()-t0:.0f}s]")
    try:
        S, matched, rA, rB = build(cid, with_spont=(ci == 0))
    except Exception as e:
        log(f"  build FAILED: {repr(e)[:150]} — skipping"); continue
    log(f"  matched A∩B n={len(matched)}")
    pooled_ids.append(matched)
    for sig in ("dff", "events"):
        data = data_for(S, rA, rB, matched, sig)
        for s in (STIM if ci == 0 else STIM[:3]):
            td = utils.extract_trials(data, s, response_window=None)
            da, dm, df = EncodingModel(td, n_basis=5).fit_all(cv="blocked").r2_decomposition()
            for term, arr in (("add", da), ("mult", dm), ("full", df)):
                POOL.setdefault((sig, s, term), []).append(np.asarray(arr))
    if ci == 0:                 # full battery on primary
        for sig in ("dff", "events"):
            data = data_for(S, rA, rB, matched, sig)
            for s in STIM:      # shuffled (leakage) + rate-ratio
                td = utils.extract_trials(data, s, response_window=None)
                _, _, dfs = EncodingModel(td, n_basis=5).fit_all(cv="shuffled").r2_decomposition()
                prim[(sig, s, "shuf_full")] = np.median(dfs)
        # per-session running (dff-based V)
        for sk in ("A", "B"):
            prim[f"V_{sk}"] = S[sk]["running_speed"][0]
        # synthetic recovery on dg (events)
        dt = data_for(S, rA, rB, matched, "events"); td = utils.extract_trials(dt, "drifting_gratings", response_window=None)
        em0 = EncodingModel(td); lab = em0._condition_labels(); R = td.responses.mean(2)
        dh = np.empty_like(R)
        for c in np.unique(lab): m = lab == c; dh[:, m] = R[:, m].mean(1, keepdims=True)
        dhc = dh - dh.mean(1, keepdims=True); vc = td.running_speed.mean(1) - td.running_speed.mean()
        prim["syn_g"] = np.array([0., .02, .05, .1, .2]); sm, sf = [], []
        for g in prim["syn_g"]:
            tds = TrialData(stimulus="dg", params=td.params, responses=td.responses + (g * vc[None, :] * dhc)[:, :, None],
                            running_speed=td.running_speed, time=td.time, stimulus_params=td.stimulus_params)
            _, dm, df = EncodingModel(tds, n_basis=5).fit_all(cv="blocked").r2_decomposition()
            sm.append(np.median(dm)); sf.append(np.median(df))
        prim["syn_mult"] = np.array(sm); prim["syn_full"] = np.array(sf)
        # paper-metric rate ratio (events) + mean diff
        for s in ("drifting_gratings", "natural_scenes"):
            td = utils.extract_trials(data_for(S, rA, rB, matched, "events"), s, response_window=None)
            Rr = td.responses.mean(2); V = td.running_speed.mean(1); run, still = V > 3, V < 0.5
            mr, ms = Rr[:, run].mean(1), Rr[:, still].mean(1); pos = ms > 0
            prim[f"ratio_{s}"] = gmean(mr[pos] / ms[pos])

# ----- pool & save -----
save = {}
for (sig, s, term), arrs in POOL.items():
    save[f"{sig}__{s}__{term}"] = np.concatenate(arrs)
for k, v in prim.items():
    save[f"prim__{k}"] = v
pooled_ids = np.concatenate(pooled_ids)
save["pooled_ids"] = pooled_ids
np.savez("data/encoding_v1.npz", **save)
n_pool = len(save["dff__drifting_gratings__full"])
log(f"\n[t+{time.time()-t0:.0f}s] pooled n={n_pool} cells across {len(CIDS)} V1 containers -> data/encoding_v1.npz")

# ----- report numbers -----
def line(sig, cv_label, stim_terms):
    pass
log(f"\n===== V1 POOLED (n={n_pool}) — blocked CV, dF/F =====")
log(f"{'stimulus':18s}| ΔR²_add             ΔR²_mult            ΔR²_full")
for s in STIM:
    a, m, f = save[f"dff__{s}__add"], save[f"dff__{s}__mult"], save[f"dff__{s}__full"]
    log(f"{s:18s}| {np.median(a):+.4f}(p={wg(a):.2g})  {np.median(m):+.4f}(p={wg(m):.2g})  {np.median(f):+.4f}(p={wg(f):.2g})")
log(f"\n===== V1 POOLED — blocked CV, EVENTS =====")
for s in STIM:
    a, m, f = save[f"events__{s}__add"], save[f"events__{s}__mult"], save[f"events__{s}__full"]
    log(f"{s:18s}| {np.median(a):+.4f}(p={wg(a):.2g})  {np.median(m):+.4f}(p={wg(m):.2g})  {np.median(f):+.4f}(p={wg(f):.2g})")

# FDR over 12 dff tests
tests = [(s, t) for s in STIM for t in ("add", "mult", "full")]
pv = np.array([wg(save[f"dff__{s}__{t}"]) for s, t in tests])
order = np.argsort(pv); thr = 0.05 * np.arange(1, 13) / 12; ok = pv[order] <= thr
kmax = np.where(ok)[0].max() if ok.any() else -1; sig_set = set(order[:kmax + 1])
log("\nFDR (BH, dff): " + ", ".join(f"{tests[i][0][:2]}-{tests[i][1]}" for i in sorted(sig_set)) if sig_set else "\nFDR: none survive")

log("\nprimary shuffled-vs-blocked full (dff):")
for s in STIM:
    log(f"  {s:18s} blocked {np.median(save[f'dff__{s}__full']):+.4f}  shuffled {prim.get(('dff',s,'shuf_full'),float('nan')):+.4f}")

# positive control: pooled ΔR²_add(dff) vs Allen run_mod
try:
    meta = pd.DataFrame.from_records(boc.get_cell_specimens()).set_index("cell_specimen_id")
    for s, col in [("drifting_gratings", "run_mod_dg"), ("static_gratings", "run_mod_sg"), ("natural_scenes", "run_mod_ns")]:
        d = save[f"dff__{s}__add"]; rm = meta.reindex(pooled_ids)[col].to_numpy(float)
        mm = np.isfinite(d) & np.isfinite(rm); rho, p = stats.spearmanr(d[mm], rm[mm])
        log(f"  posctrl ΔR²_add vs {col}: rho={rho:+.3f} p={p:.2g} (n={int(mm.sum())})")
except Exception as e:
    log("posctrl skipped:", repr(e)[:120])

log(f"\nrate ratios (events): dg={prim.get('ratio_drifting_gratings',float('nan')):.2f} ns={prim.get('ratio_natural_scenes',float('nan')):.2f}")
log(f"synthetic dg recovery (events): g={list(prim.get('syn_g',[]))} full={[round(x,4) for x in prim.get('syn_full',[])]}")
for sk in ("A", "B"):
    v = prim.get(f"V_{sk}")
    if v is not None: log(f"running session {sk}: mean={np.nanmean(v):.2f} %>3={100*np.nanmean(v>3):.1f}")
log("DONE")
