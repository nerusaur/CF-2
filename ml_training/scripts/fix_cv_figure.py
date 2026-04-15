"""
fix_cv_figure.py — standalone CV figure tuner
Tweak the parameters marked with # <-- CHANGE THIS
and run: py fix_cv_figure.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold

# ── Your actual data ───────────────────────────────────────────────────────
N_TRAIN  = 490
N_SPLITS = 5
np.random.seed(42)
y_train = np.array([0]*161 + [1]*164 + [2]*165)
np.random.shuffle(y_train)
X_dummy = np.zeros((N_TRAIN, 1))

# ── Colormaps (exact sklearn) ──────────────────────────────────────────────
cmap_cv   = plt.cm.coolwarm   # blue=train, red=validation
cmap_data = plt.cm.Paired     # class strip

# ════════════════════════════════════════════════════
# TWEAK THESE — run the file after each change
LW         = 10          # <-- bar thickness: try 8, 10, 12, 15
FIG_W      = 14          # <-- figure width in inches: try 12, 14, 16
FIG_H      = 5           # <-- figure height in inches: try 4, 5, 6
FONT_SIZE  = 10          # <-- tick label font size
TITLE_SIZE = 13          # <-- title font size
SHOW_CLASS = True        # <-- True shows class strip at bottom, False hides it
VMIN       = -0.2        # <-- color range min (sklearn default: -0.2)
VMAX       = 1.2         # <-- color range max (sklearn default: 1.2)
# ════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)

for fold, (tr, val) in enumerate(skf.split(X_dummy, y_train)):
    c = np.full(N_TRAIN, np.nan)
    c[val] = 1    # red  = validation
    c[tr]  = 0    # blue = training
    ax.scatter(
        range(N_TRAIN),
        [fold + 0.5] * N_TRAIN,
        c=c,
        marker="_",
        lw=LW,
        cmap=cmap_cv,
        vmin=VMIN, vmax=VMAX,
    )

# Class strip at bottom
if SHOW_CLASS:
    ax.scatter(
        range(N_TRAIN),
        [N_SPLITS + 0.5] * N_TRAIN,
        c=y_train,
        marker="_",
        lw=LW,
        cmap=cmap_data,
        vmin=0, vmax=2,
    )

# Y-axis
n_rows = N_SPLITS + (1 if SHOW_CLASS else 0)
ytick_labels = [f"Fold {i+1}" for i in range(N_SPLITS)]
if SHOW_CLASS:
    ytick_labels += ["Class"]

ax.set(
    yticks=np.arange(n_rows) + 0.5,
    yticklabels=ytick_labels,
    xlabel="Sample index (training pool, n=490)",
    ylabel="CV iteration",
    xlim=[0, N_TRAIN],
    ylim=[n_rows + 0.2, -0.2],
)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.set_title(
    "StratifiedKFold (n_splits=5) — ChildFocus Training Pool (n=490)\n"
    "Blue = Training  |  Red = Validation  |  Bottom = Class distribution",
    fontsize=TITLE_SIZE, fontweight="bold"
)

# Legend (outside right — Rob's style)
legend_patches = [
    Patch(color=cmap_cv(0.02),   label="Training set"),
    Patch(color=cmap_cv(0.8),    label="Validation set"),
    Patch(color=cmap_data(0.0),  label="Educational"),
    Patch(color=cmap_data(0.4),  label="Neutral"),
    Patch(color=cmap_data(0.8),  label="Overstimulating"),
]
ax.legend(handles=legend_patches, loc=(1.02, 0.55),
          fontsize=9, frameon=True)

plt.tight_layout()
fig.subplots_adjust(right=0.77)
plt.savefig("cv_figure_tweaked.png", dpi=300, bbox_inches="tight")
print("Saved: cv_figure_tweaked.png")
print("Change the values under TWEAK THESE and run again.")