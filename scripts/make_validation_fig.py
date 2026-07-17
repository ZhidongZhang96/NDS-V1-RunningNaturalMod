"""Validation figure: CV-leakage correction -> doc/figures/cv_leakage.png
Panels: (A) per-session running, (B) shuffled vs blocked ΔR² collapse,
        (C) synthetic-gain recovery on dg, (D) circular-shift null.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt

os.makedirs("doc/figures", exist_ok=True)
Z = np.load("data/encoding_cv_compare.npz", allow_pickle=True)
NULL = np.load("data/robust_null.npz", allow_pickle=True) if os.path.exists("data/robust_null.npz") else None
STIM = ["drifting_gratings", "static_gratings", "natural_scenes", "spontaneous"]
SHORT = {"drifting_gratings":"drifting\ngratings","static_gratings":"static\ngratings",
         "natural_scenes":"natural\nscenes","spontaneous":"spont."}
C_SHUF, C_BLOCK, C_DG, C_NS = "#C44E52", "#4C72B0", "#C44E52", "#55A868"
med = np.median

fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.0))

# (A) per-session running: ECDF of per-trial mean speed
axA = axes[0]
for s, c in (("drifting_gratings", C_DG), ("natural_scenes", C_NS)):
    if f"V__{s}" in Z.files:
        v = np.sort(Z[f"V__{s}"]); axA.plot(v, np.linspace(0, 1, len(v)), color=c, lw=1.8,
            label=f"{s.split('_')[0]} (Sess {'A' if s=='drifting_gratings' else 'B'}): mean {v.mean():.1f}")
axA.axvline(3, color="0.6", ls=":", lw=1); axA.set_xlim(-2, 40)
axA.set_xlabel("per-trial running speed (cm/s)"); axA.set_ylabel("cumulative fraction")
axA.set_title("A  Session A (dg) has more running than B (ns)", fontsize=10)
axA.legend(fontsize=8, loc="lower right", framealpha=1, edgecolor="0.8")

# (B) shuffled vs blocked ΔR²_full collapse
axB = axes[1]; x = np.arange(len(STIM))
sh = [med(Z[f"{s}__shuffled__full"]) for s in STIM]
bl = [med(Z[f"{s}__blocked__full"]) for s in STIM]
axB.bar(x - 0.2, sh, 0.4, color=C_SHUF, alpha=0.85, label="shuffled CV (leaky)")
axB.bar(x + 0.2, bl, 0.4, color=C_BLOCK, alpha=0.85, label="blocked CV (leak-free)")
axB.axhline(0, color="0.4", lw=0.8)
axB.set_xticks(x); axB.set_xticklabels([SHORT[s] for s in STIM], fontsize=8)
axB.set_ylabel(r"median $\Delta R^2_{\mathrm{full}}$")
axB.set_title("B  The effect collapses under leak-free CV", fontsize=10)
axB.legend(fontsize=8, framealpha=1, edgecolor="0.8")

# (C) synthetic-gain recovery on dg
axC = axes[2]; g = Z["syn_g"]
axC.plot(g, Z["syn_full"], "o-", color=C_BLOCK, lw=1.6, label=r"$\Delta R^2_{\mathrm{full}}$ recovered")
axC.plot(g, Z["syn_mult"], "s--", color="#DD8452", lw=1.4, label=r"$\Delta R^2_{\mathrm{mult}}$ recovered")
ns_full = med(Z["natural_scenes__shuffled__full"])
axC.axhline(ns_full, color=C_NS, ls=":", lw=1.5, label=f"ns shuffled level ({ns_full:+.4f})")
axC.axhline(0, color="0.4", lw=0.8)
axC.set_xlabel("injected gain g (into real dg responses)")
axC.set_ylabel(r"recovered $\Delta R^2$ (blocked CV)")
axC.set_title("C  dg CAN detect an injected ns-size gain", fontsize=10)
axC.legend(fontsize=8, framealpha=1, edgecolor="0.8", loc="upper left")

# (D) circular-shift null
axD = axes[3]
if NULL is not None:
    for s, c in (("drifting_gratings", C_DG), ("natural_scenes", C_NS)):
        f = NULL[f"{s}_shift_full"]; obs = float(NULL[f"{s}_obs_full"])
        axD.hist(f, bins=24, color=c, alpha=0.5, label=f"{s.split('_')[0]} shift-null")
        axD.axvline(obs, color=c, lw=2.2, ls="--")
    axD.axvline(0, color="0.5", lw=0.8, ls=":")
    axD.set_xlabel(r"median $\Delta R^2_{\mathrm{full}}$ (blocked CV)")
    axD.set_ylabel("shift draws")
    axD.set_title("D  Observed (dashed) sits within its shift-null", fontsize=10)
    axD.legend(fontsize=8, framealpha=1, edgecolor="0.8")
else:
    axD.text(0.5, 0.5, "circular-shift null\n(pending)", ha="center", va="center",
             transform=axD.transAxes, fontsize=10, color="0.5")
    axD.set_axis_off()

fig.tight_layout()
fig.savefig("doc/figures/cv_leakage.png", dpi=150, bbox_inches="tight")
print("wrote doc/figures/cv_leakage.png  (null=%s)" % ("yes" if NULL is not None else "PENDING"))
