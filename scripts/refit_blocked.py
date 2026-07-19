"""Canonical re-fit under leakage-free blocked CV (new default) + legacy shuffled,
   regenerating data/encoding_r2.npz (blocked) and a CV-comparison cache.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr
import utils
from utils import EncodingModel, TrialData

data = utils.load_data()
cell_ids = np.asarray(data["matched_cell_ids"])
STIM = list(utils.STIMULI)
def wg(x, alt="greater"):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0): return np.nan
    try: return wilcoxon(x, alternative=alt).pvalue
    except ValueError: return np.nan
md = np.median

TD = {s: utils.extract_trials(data, s, response_window=None) for s in STIM}
FIT = {}   # (stim, cv) -> dict
for cv in ("shuffled", "blocked"):
    for s in STIM:
        em = EncodingModel(TD[s], n_basis=5).fit_all(cv=cv)   # blocked uses default gap=5
        da, dm, df = em.r2_decomposition()
        FIT[(s, cv)] = dict(r2_null=em.r2_null, r2_add=em.r2_add, r2_mult=em.r2_mult,
                            r2_full=em.r2_full, d_add=da, d_mult=dm, d_full=df)
    print(f"  fitted all stimuli, cv={cv}", flush=True)

# ---- canonical npz = BLOCKED (same schema as before) ----
out = {"cell_ids": cell_ids}
for s in STIM:
    for k, v in FIT[(s, "blocked")].items():
        out[f"{s}__{k}"] = v
np.savez(Path("data/encoding_r2.npz"), **out)
print("saved data/encoding_r2.npz (blocked CV)")

# ---- comparison cache for the validation figure ----
cmp = {"cell_ids": cell_ids}
for (s, cv), d in FIT.items():
    for term in ("add", "mult", "full"):
        cmp[f"{s}__{cv}__{term}"] = d[f"d_{term}"]
# per-session running (per-trial mean V)
for s in STIM:
    cmp[f"V__{s}"] = TD[s].running_speed.mean(axis=1)

# ---- synthetic ns-magnitude gain injected into dg (blocked CV) ----
td_dg = TD["drifting_gratings"]; em0 = EncodingModel(td_dg); lab = em0._condition_labels()
Rdg = td_dg.responses.mean(axis=2); dhat = np.empty_like(Rdg)
for c in np.unique(lab): m = lab == c; dhat[:, m] = Rdg[:, m].mean(1, keepdims=True)
dhat_c = dhat - dhat.mean(1, keepdims=True)
vdg_c = td_dg.running_speed.mean(1) - td_dg.running_speed.mean()
gs = np.array([0.0, 0.02, 0.05, 0.1, 0.2])
syn_mult, syn_full = [], []
for g in gs:
    inj = g * vdg_c[None, :] * dhat_c
    td_syn = TrialData(stimulus="dg", params=td_dg.params,
                       responses=td_dg.responses + inj[:, :, None],
                       running_speed=td_dg.running_speed, time=td_dg.time,
                       stimulus_params=td_dg.stimulus_params)
    _, dm, df = EncodingModel(td_syn, n_basis=5).fit_all(cv="blocked").r2_decomposition()
    syn_mult.append(md(dm)); syn_full.append(md(df))
cmp["syn_g"] = gs; cmp["syn_mult"] = np.array(syn_mult); cmp["syn_full"] = np.array(syn_full)
np.savez(Path("data/encoding_cv_compare.npz"), **cmp)
print("saved data/encoding_cv_compare.npz")

# ================= REPORT (blocked CV, canonical) =================
print("\n" + "=" * 84)
print("CANONICAL RESULTS — leakage-free BLOCKED CV (gap=5)")
print("=" * 84)
print(f"{'stimulus':18s} | ΔR²_add             ΔR²_mult            ΔR²_full           | frac full>0")
for s in STIM:
    r = FIT[(s, "blocked")]
    print(f"{s:18s} | {md(r['d_add']):+.4f}(p={wg(r['d_add']):.2g})  "
          f"{md(r['d_mult']):+.4f}(p={wg(r['d_mult']):.2g})  "
          f"{md(r['d_full']):+.4f}(p={wg(r['d_full']):.2g}) | {(r['d_full']>0).mean()*100:.0f}%")

print("\n-- shuffled (legacy, leaky) for comparison --")
for s in STIM:
    r = FIT[(s, "shuffled")]
    print(f"{s:18s} | full {md(r['d_full']):+.4f}(p={wg(r['d_full']):.2g})  mult {md(r['d_mult']):+.4f}(p={wg(r['d_mult']):.2g})")

print("\n== H3 gratings vs natural (paired Wilcoxon ns>grating), BLOCKED ==")
for term in ("d_mult", "d_full"):
    for g in ("drifting_gratings", "static_gratings"):
        x, y = FIT[(g, "blocked")][term], FIT[("natural_scenes", "blocked")][term]
        try: p = wilcoxon(y, x, alternative="greater").pvalue
        except ValueError: p = np.nan
        print(f"  ΔR²_{term[2:]:4s} ns({md(y):+.4f}) > {g:18s}({md(x):+.4f})  p={p:.3g}")

print("\n== FDR (Benjamini-Hochberg) over the 12 term×stimulus tests, BLOCKED ==")
tests = [(s, t) for s in STIM for t in ("d_add", "d_mult", "d_full")]
pv = np.array([wg(FIT[(s, "blocked")][t]) for s, t in tests])
order = np.argsort(pv); m = len(pv); thresh = 0.05 * (np.arange(1, m + 1)) / m
passed = pv[order] <= thresh
kmax = np.where(passed)[0].max() if passed.any() else -1
sig = set(order[:kmax + 1]) if kmax >= 0 else set()
for i, (s, t) in enumerate(tests):
    print(f"   {'SIG ' if i in sig else '    '}{s:18s} {t[2:]:5s} p={pv[i]:.3g}")

print("\n== Positive control: ΔR²_full vs Allen run_mod_* (Spearman), BLOCKED ==")
meta = pd.read_csv("data/neurons_metadata.csv").set_index("cell_specimen_id").reindex(cell_ids)
for s, col in [("drifting_gratings","run_mod_dg"),("static_gratings","run_mod_sg"),("natural_scenes","run_mod_ns")]:
    rm, d = meta[col].to_numpy(float), FIT[(s, "blocked")]["d_full"]
    mm = np.isfinite(rm) & np.isfinite(d); rho, p = spearmanr(d[mm], rm[mm])
    print(f"  {s:18s} rho={rho:+.3f} p={p:.3g} (n={int(mm.sum())})")

print("\n== Sensitivity: responsive cells (Allen p_*<0.05), BLOCKED ==")
for s, col in [("drifting_gratings","p_dg"),("static_gratings","p_sg"),("natural_scenes","p_ns")]:
    resp = meta[col].to_numpy(float) < 0.05; d = FIT[(s, "blocked")]["d_full"][resp]
    print(f"  {s:18s} n={int(resp.sum()):2d}  median ΔR²_full={md(d):+.4f}  p={wg(d):.3g}")

print("\n== synthetic dg recovery (blocked): g, ΔR²_mult, ΔR²_full ==")
for g, sm, sf in zip(gs, syn_mult, syn_full):
    print(f"   g={g:<5}  mult={sm:+.4f}  full={sf:+.4f}")
print("\nDONE")
