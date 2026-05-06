"""
Curriculum Learning data-efficiency results — visualization.

Three figures:
  1. Aggregate test accuracy vs training-data fraction (the data-efficiency curve).
  2. Per-bin (L1..L5) accuracy across difficulty, one panel per training-data fraction.
  3. Curriculum advantage over the pacing baseline across bins and data fractions (heatmap).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----- Data ------------------------------------------------------------------
fractions = [20, 40, 60, 80, 100]
strategies = ["Pacing baseline", "Curriculum", "Anti-curriculum"]
bins = ["L1", "L2", "L3", "L4", "L5"]

# rows: (fraction, strategy) -> [L1, L2, L3, L4, L5, Aggregate]
raw = {
    (20, "Pacing baseline"):  [88.55, 73.15, 60.90, 47.90, 35.15, 61.13],
    (20, "Curriculum"):       [98.15, 93.25, 75.75, 53.25, 28.65, 69.81],
    (20, "Anti-curriculum"):  [37.30, 42.60, 43.15, 42.15, 35.25, 40.09],

    (40, "Pacing baseline"):  [95.65, 86.05, 74.35, 57.05, 38.05, 70.23],
    (40, "Curriculum"):       [98.90, 93.40, 82.60, 59.00, 31.20, 73.02],
    (40, "Anti-curriculum"):  [68.45, 64.60, 60.35, 52.95, 41.15, 57.50],

    (60, "Pacing baseline"):  [99.50, 96.90, 89.05, 70.65, 47.15, 80.65],
    (60, "Curriculum"):       [99.95, 99.40, 95.25, 77.95, 42.05, 82.92],
    (60, "Anti-curriculum"):  [91.70, 84.35, 77.40, 64.75, 45.75, 72.79],

    (80, "Pacing baseline"):  [99.75, 98.50, 92.20, 77.30, 51.55, 83.87],
    (80, "Curriculum"):       [100.00, 99.70, 96.35, 81.50, 44.60, 84.43],
    (80, "Anti-curriculum"):  [97.55, 92.20, 83.90, 69.40, 49.25, 78.46],

    (100, "Pacing baseline"): [99.90, 98.50, 92.55, 77.65, 52.90, 84.30],
    (100, "Curriculum"):      [100.00, 99.50, 96.75, 83.65, 50.65, 86.11],
    (100, "Anti-curriculum"): [99.15, 94.60, 86.60, 70.60, 48.30, 79.85],
}

# Tidy DataFrame for convenience
records = []
for (f, s), vals in raw.items():
    rec = {"fraction": f, "strategy": s,
           "L1": vals[0], "L2": vals[1], "L3": vals[2],
           "L4": vals[3], "L5": vals[4], "Aggregate": vals[5]}
    records.append(rec)
df = pd.DataFrame(records).sort_values(["fraction", "strategy"]).reset_index(drop=True)
print("=== Tidy results table ===")
print(df.to_string(index=False))

# Consistent styling
colors = {
    "Pacing baseline":  "#4C78A8",   # blue
    "Curriculum":       "#2CA02C",   # green
    "Anti-curriculum":  "#D62728",   # red
}
markers = {"Pacing baseline": "o", "Curriculum": "s", "Anti-curriculum": "^"}

# ----- Figure 1: aggregate accuracy vs training-data fraction ---------------
fig1, ax = plt.subplots(figsize=(8, 5.5))
for s in strategies:
    sub = df[df.strategy == s].sort_values("fraction")
    ax.plot(sub.fraction, sub.Aggregate,
            color=colors[s], marker=markers[s], markersize=8, linewidth=2.2, label=s)
    for x, y in zip(sub.fraction, sub.Aggregate):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=colors[s])

ax.set_xlabel("Training-data fraction (%)")
ax.set_ylabel("Aggregate test accuracy (%)")
ax.set_title("Data efficiency: aggregate test accuracy vs training-data fraction")
ax.set_xticks(fractions)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", frameon=True)
ax.set_ylim(35, 92)
fig1.tight_layout()
fig1.savefig("/mnt/user-data/outputs/fig1_aggregate_accuracy.png", dpi=160)

# ----- Figure 2: per-bin accuracy, one panel per training-data fraction -----
fig2, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
x_idx = np.arange(len(bins))
for ax, f in zip(axes, fractions):
    for s in strategies:
        vals = raw[(f, s)][:5]
        ax.plot(x_idx, vals, color=colors[s], marker=markers[s],
                markersize=7, linewidth=2, label=s)
    ax.set_title(f"{f}% training data")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Test bin (easy → hard)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(20, 102)
axes[0].set_ylabel("Test accuracy (%)")
axes[-1].legend(loc="lower left", frameon=True, fontsize=9)
fig2.suptitle("Per-bin test accuracy across difficulty (L1 = easiest, L5 = hardest)",
              fontsize=13, y=1.02)
fig2.tight_layout()
fig2.savefig("/mnt/user-data/outputs/fig2_per_bin_accuracy.png", dpi=160, bbox_inches="tight")

# ----- Figure 3: curriculum advantage over pacing baseline (heatmap) --------
adv_curr = np.zeros((len(fractions), len(bins)))
adv_anti = np.zeros((len(fractions), len(bins)))
for i, f in enumerate(fractions):
    base = np.array(raw[(f, "Pacing baseline")][:5])
    adv_curr[i] = np.array(raw[(f, "Curriculum")][:5]) - base
    adv_anti[i] = np.array(raw[(f, "Anti-curriculum")][:5]) - base

fig3, axes = plt.subplots(1, 2, figsize=(13, 4.8))
vmax = max(np.abs(adv_curr).max(), np.abs(adv_anti).max())
for ax, M, title in zip(axes, [adv_curr, adv_anti],
                        ["Curriculum − Pacing baseline", "Anti-curriculum − Pacing baseline"]):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins)
    ax.set_yticks(range(len(fractions))); ax.set_yticklabels([f"{f}%" for f in fractions])
    ax.set_xlabel("Test bin")
    ax.set_ylabel("Training-data fraction")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.55 else "black", fontsize=9)
    fig3.colorbar(im, ax=ax, label="Δ accuracy (pp)")
fig3.suptitle("Strategy advantage over the pacing baseline (percentage points)",
              fontsize=13, y=1.02)
fig3.tight_layout()
fig3.savefig("/mnt/user-data/outputs/fig3_advantage_heatmap.png", dpi=160, bbox_inches="tight")

# ----- Quick summary numbers -----------------------------------------------
print("\n=== Aggregate accuracy summary ===")
agg = df.pivot(index="fraction", columns="strategy", values="Aggregate")[strategies]
print(agg.round(2))

print("\n=== Curriculum advantage over baseline (aggregate, pp) ===")
print((agg["Curriculum"] - agg["Pacing baseline"]).round(2))

print("\n=== Anti-curriculum advantage over baseline (aggregate, pp) ===")
print((agg["Anti-curriculum"] - agg["Pacing baseline"]).round(2))

print("\nSaved: fig1_aggregate_accuracy.png, fig2_per_bin_accuracy.png, fig3_advantage_heatmap.png")
