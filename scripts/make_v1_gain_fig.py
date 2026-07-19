"""Figure 1 for the V1-centric report: population running gain, V1 vs VISpm.
Medians from scripts/compute_v1_metrics.py (V1 pooled n=363) and the VISpm runs (n=47),
on Allen L0 events (spike-comparable). -> doc/figures/v1_gain.png
"""
import os
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt

os.makedirs("doc/figures", exist_ok=True)
STIM = ["drifting\ngratings", "static\ngratings", "natural\nscenes"]
# events (spike-comparable) medians
V1_ratio = [1.57, 1.72, 2.15];   VISpm_ratio = [1.04, 1.04, 1.10]
V1_gain  = [0.34, 0.47, 0.77];   VISpm_gain  = [0.08, 0.09, 0.36]
C_V1, C_PM = "#C44E52", "#4C72B0"
x = np.arange(3)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

a1.bar(x - 0.2, V1_ratio, 0.4, color=C_V1, label="V1 (VISp), n=363")
a1.bar(x + 0.2, VISpm_ratio, 0.4, color=C_PM, label="VISpm, n=47")
a1.axhline(1.0, color="0.4", lw=0.9, ls="-")
a1.axhline(1.40, color="0.4", lw=1.1, ls="--")
a1.text(2.45, 1.42, "Liska/Yates\nV1 = 1.40", fontsize=8, color="0.35", ha="right", va="bottom")
a1.set_xticks(x); a1.set_xticklabels(STIM)
a1.set_ylabel("running / stationary rate ratio")
a1.set_title("A  Population running gain (events)", fontsize=11)
a1.legend(fontsize=9, framealpha=1, edgecolor="0.8")

a2.bar(x - 0.2, V1_gain, 0.4, color=C_V1, label="V1 (VISp)")
a2.bar(x + 0.2, VISpm_gain, 0.4, color=C_PM, label="VISpm")
a2.axhline(1.0, color="0.4", lw=0.9, ls="--")
a2.text(2.45, 1.02, "no gain", fontsize=8, color="0.35", ha="right", va="bottom")
a2.set_xticks(x); a2.set_xticklabels(STIM)
a2.set_ylabel("tuning-gain slope (running vs still tuning)")
a2.set_title("B  Multiplicative gain (events)", fontsize=11)
a2.set_ylim(0, 1.1)
a2.legend(fontsize=9, framealpha=1, edgecolor="0.8")

fig.suptitle("V1 shows strong, area-specific running gain that VISpm lacks — "
             "yet single-trial cross-validated ΔR² is ~null for both (see §7.2–7.3)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig("doc/figures/v1_gain.png", dpi=150, bbox_inches="tight")
print("wrote doc/figures/v1_gain.png")
