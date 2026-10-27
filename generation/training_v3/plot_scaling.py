"""Scaling figure: 95th-percentile test error of an emulator vs training-set size, and vs multipole.
(a) median over multipoles, (b) worst multipole, (c) error vs multipole for each training-set size."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

OBS = sys.argv[1] if len(sys.argv) > 1 else "TT"
R = "/home/cmbagent/rds/ede-v3/runs"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6a65"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]      # light -> dark: 25k ... 397k

def pct(x): return 100 * (10 ** np.asarray(x) - 1)

YT = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
def yticks(ax):
    ax.yaxis.set_major_locator(FixedLocator(YT)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))

sizes = ("25000", "50000", "100000", "200000", "all")
narrow = [json.load(open(f"{R}/cs_{OBS}_N{n}_h512x512x512x512_pca0_eval.json")) for n in sizes]
wide = json.load(open(f"{R}/cs_{OBS}_Nall_h1024x1024x1024x1024_pca0_eval.json"))
v2 = json.load(open(f"{R}/v2sub_{OBS}_eval.json"))
N = np.array([r["ntrain"] for r in narrow])
p95 = [np.asarray(r["per_mode"]["95"]) for r in narrow]
ell = np.arange(2, 2 + len(v2["95"]))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.45]})

for ax, stat, title, off in ((axes[0], np.median, "Typical multipole (median over $\\ell$)", (-10, 0, "right", "center")),
                             (axes[1], np.max, "Worst multipole (maximum over $\\ell$)", (0, -12, "center", "top"))):
    y = [pct(stat(p)) for p in p95]
    yv2, yw = pct(stat(v2["95"])), pct(stat(wide["per_mode"]["95"]))
    ax.plot(N, y, "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.2, zorder=3)
    ax.plot(wide["ntrain"], yw, "D", color=ORANGE, ms=9, mec="white", mew=1.2, zorder=4)
    ax.axhline(yv2, color=GREY, lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.text(N[-1] * 1.05, yv2 * 1.07, "v2 (4x512, 250k)", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.annotate("v3, 4x512", (N[0], y[0]), xytext=(9, 2), textcoords="offset points", color=INK, fontsize=9,
                fontweight="bold", va="bottom")
    ax.annotate("v3, 4x1024", (wide["ntrain"], yw), xytext=off[:2], textcoords="offset points",
                color=INK, fontsize=9, fontweight="bold", ha=off[2], va=off[3])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training spectra")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(which="major", color=GRID, lw=0.6); ax.grid(which="minor", color=GRID, lw=0.3, alpha=0.6)
    ax.set_xticks([25000, 50000, 100000, 200000, 400000], ["25k", "50k", "100k", "200k", "400k"])
    ax.minorticks_off()
    lo = min(min(y), yw, yv2); hi = max(max(y), yv2)
    ax.set_ylim(lo / 1.5, hi * 1.5); yticks(ax)
axes[0].set_ylabel(r"95th percentile of $|C_\ell^{\rm emu}/C_\ell^{\rm CLASS}-1|$  [%]")

ax = axes[2]
for i, (r, p) in enumerate(zip(narrow, p95)):
    ax.loglog(ell, pct(p), color=SEQ[i], lw=1.1, label=f"v3, 4x512, {r['ntrain'] // 1000}k")
ax.loglog(ell, pct(wide["per_mode"]["95"]), color=ORANGE, lw=1.6, label=f"v3, 4x1024, {wide['ntrain'] // 1000}k")
ax.loglog(ell, pct(v2["95"]), color=GREY, lw=1.1, ls=(0, (5, 3)), label="v2, 4x512, 250k")
ax.set_xlabel(r"Multipole $\ell$")
ax.set_title("95th percentile by multipole", fontsize=11, color=INK, loc="left")
ax.grid(which="major", color=GRID, lw=0.6); ax.grid(which="minor", color=GRID, lw=0.3, alpha=0.6)
ax.set_xlim(2, ell[-1])
ax.set_ylim(0.012, 1.2); yticks(ax)
ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Emulator, training spectra",
          title_fontsize=8.5, alignment="left")

for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle(f"{OBS} emulator accuracy vs training-set size  (CLASS-EDE test set, 99,216 spectra)", fontsize=12, color=INK, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(f"/home/cmbagent/rds/ede-v3/scaling_{OBS}_clean.png", dpi=300, facecolor="white")
print("saved")
