"""Demonstration figure: the EncodingModel pipeline + design components + the model
equation (LaTeX), on real data (example neuron, drifting gratings).
-> doc/figures/pipeline_demo.png
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import utils
from utils import EncodingModel

os.makedirs("doc/figures", exist_ok=True)
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

C1, C2, C3, C4 = "#4C72B0", "#55A868", "#DD8452", "#C44E52"
fig = plt.figure(figsize=(13.5, 6.4))
gs = GridSpec(2, 4, height_ratios=[1.05, 0.8], hspace=0.55, wspace=0.34,
              left=0.055, right=0.985, top=0.9, bottom=0.13)

# ---------------- Top: pipeline schematic + LaTeX model equation ----------------
axs = fig.add_subplot(gs[0, :]); axs.axis("off"); axs.set_xlim(0, 100); axs.set_ylim(0, 10)
boxes = [
    ("data", "ΔF/F · running\n· stim tables", "#EEF2F7"),
    ("extract_trials", "TrialData:\nresp · V · params", "#E7EEF6"),
    ("_build_design", "f(S) · φ(t)\nV · V·d̂(S)", "#E3F0E8"),
    ("4 nested models", "Null·Add\nMult·Full", "#FBEDE2"),
    ("fit_all", "blocked-CV\nridge (GCV λ)", "#F6E7E8"),
    ("outputs", "ΔR² decomp.\n+ pop. gain", "#EEF2F7"),
]
xw, gap = 14.8, 1.35
for i, (title, sub, fc) in enumerate(boxes):
    x = 0.6 + i * (xw + gap)
    axs.add_patch(FancyBboxPatch((x, 6.0), xw, 3.6, boxstyle="round,pad=0.12,rounding_size=0.4",
                                 fc=fc, ec="#5A6B7B", lw=1.2))
    axs.text(x + xw / 2, 8.6, title, ha="center", va="center", fontsize=9, fontweight="bold")
    axs.text(x + xw / 2, 7.0, sub, ha="center", va="center", fontsize=7.2, color="#333")
    if i < len(boxes) - 1:
        axs.add_patch(FancyArrowPatch((x + xw, 7.8), (x + xw + gap, 7.8),
                                      arrowstyle="-|>", mutation_scale=12, lw=1.5, color="#5A6B7B"))

# main model equation (Full model), LaTeX via mathtext
eq = (r"$r_i(t) = A_i\,s(t) + \beta_0 + \sum_j b_{ij}\,\phi_j(t)"
      r" + \beta_{\mathrm{add}}\,V(t) + \beta_{\mathrm{mult}}\,[\,V(t)\,\hat{d}_i(S)\,]$")
axs.text(50, 3.7, eq, ha="center", va="center", fontsize=15.5)
axs.text(50, 1.55,
         r"② tuning $A_i s(t)$    ① drift $\sum_j b_{ij}\phi_j$    "
         r"③ running $\beta_{\mathrm{add}}V$    ④ gain $\beta_{\mathrm{mult}}(V\hat{d})$",
         ha="center", va="center", fontsize=9, color="#555")
axs.text(50, 0.3,
         r"$\Delta R^2_x = R^2_x - R^2_{\mathrm{null}}$  per cell (cross-validated);  "
         r"nested: Null $\subset$ Add, Mult $\subset$ Full",
         ha="center", va="center", fontsize=9, color="#444")

# ---------------- Bottom: real design components (example cell), compressed ----------------
a = fig.add_subplot(gs[1, 0])
for j in range(drift.shape[1]):
    a.plot(tvec, drift[:, j], lw=1.3)
a.plot(tvec, drift.sum(1), "k--", lw=0.9, alpha=0.6, label=r"$\Sigma\phi=1$")
a.set_title("① drift basis  $\\phi_j(t)$", fontsize=9.5)
a.set_xlabel("trial time (s)", fontsize=8); a.set_ylabel("basis", fontsize=8)
a.legend(fontsize=7, loc="center right"); a.tick_params(labelsize=7)

b = fig.add_subplot(gs[1, 1])
b.bar(np.arange(len(oris)), ori_tuning, color=C2, width=0.72)
b.set_xticks(np.arange(len(oris))); b.set_xticklabels([f"{int(o)}" for o in oris], fontsize=7)
b.set_title(f"② tuning  $f(S)=A_i s(t)$\n(cell #{ex})", fontsize=9.5)
b.set_xlabel("orientation (°)", fontsize=8); b.set_ylabel("mean ΔF/F", fontsize=8)
b.axhline(0, color="0.6", lw=0.6); b.tick_params(labelsize=7)

n = min(140, len(V))
c = fig.add_subplot(gs[1, 2])
c.plot(np.arange(n), V[:n], color=C3, lw=0.9)
c.axhline(0, color="0.6", lw=0.6)
c.set_title("③ running  $V(t)$", fontsize=9.5)
c.set_xlabel("trial", fontsize=8); c.set_ylabel("cm/s", fontsize=8); c.tick_params(labelsize=7)

d = fig.add_subplot(gs[1, 3])
d.plot(np.arange(n), (V * dhat[ex])[:n], color=C4, lw=0.9)
d.axhline(0, color="0.6", lw=0.6)
d.set_title(r"④ interaction  $V(t)\,\hat{d}_i(S)$", fontsize=9.5)
d.set_xlabel("trial", fontsize=8); d.set_ylabel(r"$V\cdot\hat{d}$", fontsize=8); d.tick_params(labelsize=7)

fig.suptitle("EncodingModel — pipeline, model equation, and design components (example neuron)",
             fontsize=12.5, fontweight="bold", y=0.98)
fig.savefig("doc/figures/pipeline_demo.png", dpi=150, bbox_inches="tight")
print(f"wrote doc/figures/pipeline_demo.png  (cell #{ex}, {R.shape[1]} trials, {len(conds)} conditions)")
