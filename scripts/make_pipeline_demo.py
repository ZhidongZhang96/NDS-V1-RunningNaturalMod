"""Demonstration figure (color-coded form): pipeline + color-coded model equation + the REAL
design components (tent-basis drift, tuning, running, interaction) on real data.
-> doc/figures/pipeline_demo.png
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
from matplotlib.gridspec import GridSpec
import utils
from utils import EncodingModel

os.makedirs("doc/figures", exist_ok=True)

# term colors (equation term <-> panel title <-> panel data)
GREEN, BLUE, GOLD, DARKRED, BLACK = "#2E8B3D", "#2C6FB4", "#BE8A1E", "#8E3A3A", "#1a1a1a"

data = utils.load_data()
td = utils.extract_trials(data, "drifting_gratings", response_window=None)
em = EncodingModel(td, n_basis=5)
R = em._trial_response(); labels = em._condition_labels()
V = em._running_speed(); drift = em._drift_basis(); dhat = em._stimulus_mean()
tvec = td.time.mean(1); ori = np.asarray(td.stimulus_params["orientation"], float)
conds = np.unique(labels)
tun = np.array([[R[c, labels == k].mean() for k in conds] for c in range(R.shape[0])])
ex = int(np.argmax(tun.max(1) - tun.min(1)))
oris = np.unique(ori[np.isfinite(ori)])
ori_tuning = np.array([R[ex, ori == o].mean() for o in oris])

fig = plt.figure(figsize=(13.6, 5.3))
gs = GridSpec(3, 4, height_ratios=[0.46, 0.15, 0.85], hspace=0.16, wspace=0.33,
              left=0.055, right=0.985, top=0.945, bottom=0.12)

# ---------------- Band 1: pipeline schematic (boxes) ----------------
axs = fig.add_subplot(gs[0, :]); axs.axis("off"); axs.set_xlim(0, 100); axs.set_ylim(0, 3)
boxes = [
    ("data", "raw data", "#EEF2F7"),
    ("extract_trials", "per-stimulus trials", "#E7EEF6"),
    ("_build_design", "assemble regressors", "#E3F0E8"),
    ("4 nested models", "Null, Add, Mult, Full", "#FBEDE2"),
    ("fit_all", "ridge regression\nwith block CV", "#F6E7E8"),
    ("outputs", "cross-validated $R^2$", "#EFE9F3"),
]
xw, gap = 15.3, 1.0
for i, (title, sub, fc) in enumerate(boxes):
    x = 1.6 + i * (xw + gap)
    axs.add_patch(FancyBboxPatch((x, 0.35), xw, 2.35, boxstyle="round,pad=0.08,rounding_size=0.3",
                                 fc=fc, ec="#5A6B7B", lw=1.2))
    axs.text(x + xw / 2, 1.95, title, ha="center", va="center", fontsize=13.5, fontweight="bold")
    axs.text(x + xw / 2, 0.9, sub, ha="center", va="center", fontsize=10.5, color="#333")
    if i < len(boxes) - 1:
        axs.add_patch(FancyArrowPatch((x + xw, 1.5), (x + xw + gap, 1.5),
                                      arrowstyle="-|>", mutation_scale=11, lw=1.4, color="#5A6B7B"))

# ---------------- Band 2: color-coded model equation (HPacker of colored segments) ----------------
axe = fig.add_subplot(gs[1, :]); axe.axis("off"); axe.set_xlim(0, 1); axe.set_ylim(0, 1)
segs = [(r"$r_i(t) = $", BLACK), (r"$A_i\,s(t)$", GREEN), (r"$ + \beta_0 + $", BLACK),
        (r"$\sum_j b_{ij}\,\phi_j(t)$", BLUE), (r"$ + $", BLACK),
        (r"$\beta_{\mathrm{add}}\,V(t)$", GOLD), (r"$ + $", BLACK),
        (r"$\beta_{\mathrm{mult}}\,[\,V(t)\,\hat{d}_i(S)\,]$", DARKRED)]
children = [TextArea(t, textprops=dict(color=c, fontsize=16.5)) for t, c in segs]
eq = HPacker(children=children, align="baseline", pad=0, sep=3)
axe.add_artist(AnnotationBbox(eq, (0.5, 0.5), xycoords=axe.transAxes, frameon=False,
                              box_alignment=(0.5, 0.5), pad=0))

# ---------------- Bottom: REAL design components (corrected) ----------------
blues = plt.cm.Blues(np.linspace(0.45, 0.9, drift.shape[1]))
a = fig.add_subplot(gs[2, 0])
for j in range(drift.shape[1]):
    a.plot(tvec, drift[:, j], lw=1.4, color=blues[j])
a.plot(tvec, drift.sum(1), "k--", lw=0.9, alpha=0.6, label=r"$\Sigma\phi=1$")
a.set_title(r"Drift Basis: $\sum_j b_{ij}\phi_j(t)$", fontsize=10, color=BLUE)
a.set_xlabel("trial time (s)", fontsize=8); a.set_ylabel("tent basis value", fontsize=8)
a.legend(fontsize=7, loc="center right"); a.tick_params(labelsize=7)

b = fig.add_subplot(gs[2, 1])
b.bar(np.arange(len(oris)), ori_tuning, color=GREEN, width=0.72)
b.set_xticks(np.arange(len(oris))); b.set_xticklabels([f"{int(o)}" for o in oris], fontsize=7)
b.set_title(r"Tuning: $A_i s(t) + \beta_0$", fontsize=10, color=GREEN)
b.set_xlabel("orientation (°)", fontsize=8); b.set_ylabel("mean ΔF/F", fontsize=8)
b.axhline(0, color="0.6", lw=0.6); b.tick_params(labelsize=7)

n = min(140, len(V))
c = fig.add_subplot(gs[2, 2])
c.plot(np.arange(n), V[:n], color=GOLD, lw=0.9)
c.axhline(0, color="0.6", lw=0.6)
c.set_title(r"Running: $\beta_{\mathrm{add}} V(t)$", fontsize=10, color=GOLD)
c.set_xlabel("trial", fontsize=8); c.set_ylabel("running speed (cm/s)", fontsize=8); c.tick_params(labelsize=7)

d = fig.add_subplot(gs[2, 3])
d.plot(np.arange(n), (V * dhat[ex])[:n], color=DARKRED, lw=0.9)
d.axhline(0, color="0.6", lw=0.6)
d.set_title(r"Interaction: $\beta_{\mathrm{mult}}[V(t)\hat{d}_i(S)]$", fontsize=10, color=DARKRED)
d.set_xlabel("trial", fontsize=8); d.set_ylabel(r"$V\cdot\hat{d}$", fontsize=8); d.tick_params(labelsize=7)

fig.suptitle("EncodingModel — pipeline, model equation, and design components (example neuron)",
             fontsize=12.5, fontweight="bold", y=0.985)
fig.savefig("doc/figures/pipeline_demo.png", dpi=150, bbox_inches="tight")
print(f"wrote doc/figures/pipeline_demo.png  (cell #{ex}, {R.shape[1]} trials, {len(conds)} conditions)")
