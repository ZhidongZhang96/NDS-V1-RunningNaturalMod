import warnings; warnings.filterwarnings("ignore")
import os
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

os.makedirs("doc/figures", exist_ok=True)
Z = np.load("data/encoding_r2.npz", allow_pickle=True)
cell_ids = Z["cell_ids"]
STIM = ["drifting_gratings", "static_gratings", "natural_scenes", "spontaneous"]
SHORT = {"drifting_gratings": "drifting\ngratings", "static_gratings": "static\ngratings",
         "natural_scenes": "natural\nscenes", "spontaneous": "spont."}
TERMS = [("add", r"$\Delta R^2_{\mathrm{add}}$", "#4C72B0"),
         ("mult", r"$\Delta R^2_{\mathrm{mult}}$", "#DD8452"),
         ("full", r"$\Delta R^2_{\mathrm{full}}$", "#55A868")]

def delta(stim, term):
    return np.asarray(Z[f"{stim}__d_{term}"], dtype=float)

def median_ci(x, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.median(x), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

def wg(x):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0): return 1.0
    try: return wilcoxon(x, alternative="greater").pvalue
    except ValueError: return 1.0

# ===== Figure 1: grouped median ΔR² ± 95% CI by stimulus (blocked CV) =====
fig, ax = plt.subplots(figsize=(8, 4.2))
xpos = np.arange(len(STIM))
dx = {"add": -0.24, "mult": 0.0, "full": 0.24}
for term, tlabel, color in TERMS:
    meds, lo, hi, filled = [], [], [], []
    for stim in STIM:
        d = delta(stim, term)
        m, l, h = median_ci(d)
        meds.append(m); lo.append(m - l); hi.append(h - m); filled.append(wg(d) < 0.05)
    xs = xpos + dx[term]
    ax.errorbar(xs, meds, yerr=[lo, hi], fmt="none", ecolor=color, capsize=3, lw=1.3, zorder=2)
    fc = [color if f else "white" for f in filled]
    ax.scatter(xs, meds, s=44, facecolors=fc, edgecolors=color, linewidths=1.5, zorder=3, label=tlabel)
ax.axhline(0, color="0.55", lw=0.9, ls="--", zorder=1)
ax.set_xticks(xpos); ax.set_xticklabels([SHORT[s] for s in STIM])
ax.set_ylabel("median $\\Delta R^2$ (cross-validated)\n$\\pm$95% bootstrap CI")
ax.set_title("Running-speed $\\Delta R^2$ decomposition across stimuli (leakage-free blocked CV)")
ax.legend(title="term  (filled = Wilcoxon $p<0.05$)", frameon=True, fontsize=9,
          title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          framealpha=1, edgecolor="0.7")
fig.tight_layout()
fig.savefig("doc/figures/dR2_decomposition.png", dpi=150, bbox_inches="tight")
print("wrote doc/figures/dR2_decomposition.png")

# ===== Figure 2: positive control scatter (ΔR²_full vs Allen run_mod_*) =====
meta = pd.read_csv("data/neurons_metadata.csv").set_index("cell_specimen_id").reindex(cell_ids)
pairs = [("drifting_gratings", "run_mod_dg"), ("static_gratings", "run_mod_sg"),
         ("natural_scenes", "run_mod_ns")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for ax, (stim, col) in zip(axes, pairs):
    d = delta(stim, "full"); rm = meta[col].to_numpy(float)
    m = np.isfinite(d) & np.isfinite(rm)
    ax.scatter(rm[m], d[m], s=16, alpha=0.6, color="#333", edgecolors="none")
    rho, p = spearmanr(d[m], rm[m])
    b, a = np.polyfit(rm[m], d[m], 1)
    xr = np.array([rm[m].min(), rm[m].max()])
    ax.plot(xr, a + b * xr, color="#C44", lw=1.4)
    ax.axhline(0, color="0.75", lw=0.7, ls="--"); ax.axvline(0, color="0.75", lw=0.7, ls="--")
    ax.set_title(f"{stim.replace('_', ' ')}\n$\\rho$={rho:+.2f}, p={p:.2f} (n={int(m.sum())})", fontsize=9.5)
    ax.set_xlabel(f"Allen {col}")
axes[0].set_ylabel("$\\Delta R^2_{\\mathrm{full}}$ (encoding model)")
fig.suptitle("Positive control: encoding-model $\\Delta R^2_{\\mathrm{full}}$ vs Allen running-modulation index",
             y=1.03, fontsize=11)
fig.tight_layout()
fig.savefig("doc/figures/validation_runmod.png", dpi=150, bbox_inches="tight")
print("wrote doc/figures/validation_runmod.png")
print("DONE")
