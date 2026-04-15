"""
Step 4b — Hyperparameter Tuning: TF-IDF grid search for Complement NB
ChildFocus — ml_training/scripts/4b_tune_tfidf.py

Sweeps max_features and min_df of the TF-IDF vectorizer.
Uses BEST_ALPHA = 1.5 from Step 4 (4tunealpha.py).
CV-only — test set (test_210.csv) still SEALED.

Four metrics tracked per configuration:
  Accuracy  — primary selection metric (replicated from Step 4)
  F1-Score  — tiebreaker; also confirms accuracy gain is real
  Precision — reported for completeness
  Recall    — reported for completeness

Best config selected by highest CV accuracy, F1 as tiebreaker.

Run from ml_training/scripts/:
    py 4b_tune_tfidf.py

Produces 4 figures saved to ../outputs/figures/:
    fig1_stratified_kfold_split.png   — fold structure
    fig2_gridsearch_heatmap.png       — CV mean & std heatmap
    fig3_accuracy_errorbar.png        — mean +/- std all 16 configs
    fig4_best_config_folds.png        — per-fold accuracy of winner

After this: set BEST_MAX_FEATURES and BEST_MIN_DF in 5final_eval.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── Output directory ──────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

TRAIN_COLOR = "#3274A1"
TEST_COLOR  = "#E1812C"
CLASS_CMAP  = plt.cm.Set2

# ── Best alpha from Step 4 ─────────────────────────────────────────────────
BEST_ALPHA = 1.5

# ── Text formula ──────────────────────────────────────────────────────────
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","this","that","was","are","be","as","so","we",
    "he","she","they","you","i","my","your","his","her","its","our","their",
    "what","which","who","will","would","could","should","has","have","had",
    "do","does","did","not","no","if","then","than","when","where","how",
    "all","each","more","also","just","can","up","out","about","into",
    "too","very","s","t","re","ve","ll","d",
}

def build_nb_text(title="", tags=None, description=""):
    title_part = f"{title} " * 3
    if isinstance(tags, list):
        tags_str = " ".join(str(t) for t in tags)
    else:
        tags_str = str(tags) if tags and str(tags) != "nan" else ""
    desc_part = str(description or "")[:300]
    raw = f"{title_part}{tags_str} {desc_part}".lower()
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    raw = re.sub(r"[^a-z\s]", " ", raw)
    tokens = [t for t in raw.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

# ── Load train_490.csv ────────────────────────────────────────────────────
LABELS = ["Educational", "Neutral", "Overstimulating"]

df   = pd.read_csv("train_490.csv")
df   = df[df["label"].isin(LABELS)].reset_index(drop=True)
X_tr = df.apply(lambda r: build_nb_text(
    title=r.get("title", ""),
    tags=r.get("tags", ""),
    description=r.get("description", "")
), axis=1).tolist()
y_tr = df["label"].tolist()

print(f"Training samples loaded: {len(X_tr)}  <- must be 490")
assert len(X_tr) == 490, f"Expected 490, got {len(X_tr)}"
print(f"Alpha fixed at         : {BEST_ALPHA}  (from Step 4)")
print(f"Test set               : still SEALED\n")

le    = LabelEncoder()
y_enc = le.fit_transform(y_tr)

# ── 4 metrics scoring dict ────────────────────────────────────────────────
scoring = {
    "accuracy":  "accuracy",
    "precision": "precision_macro",
    "recall":    "recall_macro",
    "f1":        "f1_macro",
}

# =============================================================================
# FIGURE 1 — Stratified K-Fold Split Visualization
# =============================================================================
print("Generating Figure 1: Stratified K-Fold split visualization...")

N_SPLITS = 5
skf_viz  = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
cmap_cv  = plt.cm.RdYlBu_r

fig1, ax1 = plt.subplots(figsize=(12, 5))
fold_train_n = []
fold_test_n  = []

for fold_i, (tr_idx, te_idx) in enumerate(skf_viz.split(X_tr, y_enc)):
    indices         = np.full(len(X_tr), np.nan)
    indices[te_idx] = 1.0
    indices[tr_idx] = 0.0
    ax1.scatter(range(len(indices)), [fold_i + 0.5] * len(indices),
                c=indices, marker="|", lw=10, cmap=cmap_cv,
                vmin=-0.2, vmax=1.2, s=10)
    fold_train_n.append(len(tr_idx))
    fold_test_n.append(len(te_idx))

ax1.scatter(range(len(y_enc)), [N_SPLITS + 0.5] * len(y_enc),
            c=y_enc, marker="|", lw=10, cmap=CLASS_CMAP, vmin=0, vmax=2, s=10)

ytick_labels = [
    f"Fold {i+1}   train={fold_train_n[i]}  val={fold_test_n[i]}"
    for i in range(N_SPLITS)
] + ["Class label"]

ax1.set(
    yticks=np.arange(N_SPLITS + 1) + 0.5,
    yticklabels=ytick_labels,
    xlabel="Sample index  (train_490.csv, sorted by original order)",
    xlim=[0, len(X_tr)],
    ylim=[N_SPLITS + 1.2, -0.2],
)
ax1.set_title(
    f"StratifiedKFold (n_splits={N_SPLITS}) — ChildFocus NB Training Pool  "
    f"(n=490, 3 classes)",
    fontsize=12, fontweight="bold", pad=10
)
ax1.tick_params(axis="y", labelsize=9)
train_p = mpatches.Patch(color=cmap_cv(0.05), label="Training samples")
val_p   = mpatches.Patch(color=cmap_cv(0.95), label="Validation samples")
e_p     = mpatches.Patch(color=CLASS_CMAP(0.0), label=f"Educational  (n={int(np.sum(y_enc==0))})")
n_p     = mpatches.Patch(color=CLASS_CMAP(0.4), label=f"Neutral       (n={int(np.sum(y_enc==1))})")
o_p     = mpatches.Patch(color=CLASS_CMAP(0.7), label=f"Overstimulating (n={int(np.sum(y_enc==2))})")
ax1.legend(handles=[train_p, val_p, e_p, n_p, o_p],
           loc="upper right", fontsize=9, ncol=2,
           title="Split  |  Class label", framealpha=0.9)
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "fig1_stratified_kfold_split.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved -> {p1}")


# =============================================================================
# GRID SEARCH — 16 configurations, all 4 metrics
# =============================================================================
max_features_opts = [5000, 10000, 15000, 20000]
min_df_opts       = [1, 2, 3, 5]

skf              = StratifiedKFold(n_splits=5, shuffle=False)
rows             = []
means            = np.zeros((len(max_features_opts), len(min_df_opts)))
stds             = np.zeros((len(max_features_opts), len(min_df_opts)))
best_mean        = 0.0
best_f1_gs       = 0.0
best_cfg         = None
best_fold_scores = None

# Step 4 baseline for reference
baseline_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000,
                              sublinear_tf=True, min_df=2, strip_accents="unicode")),
    ("clf",   ComplementNB(alpha=BEST_ALPHA))
])
baseline_cv = cross_val_score(baseline_pipe, X_tr, y_tr, cv=skf, scoring="accuracy")
baseline    = baseline_cv.mean() * 100

print("=" * 88)
print(f"  {'max_feat':>10}  {'min_df':>7}  {'Acc%':>8}  {'±Acc':>7}  "
      f"{'Prec%':>7}  {'Rec%':>7}  {'F1':>8}")
print("=" * 88)

prev_mf = None
for i, mf in enumerate(max_features_opts):
    for j, md in enumerate(min_df_opts):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=mf, min_df=md,
                sublinear_tf=True, strip_accents="unicode"
            )),
            ("clf", ComplementNB(alpha=BEST_ALPHA))
        ])
        cv   = cross_validate(pipe, X_tr, y_tr, cv=skf, scoring=scoring)
        mean = cv["test_accuracy"].mean()
        std  = cv["test_accuracy"].std()
        prec = cv["test_precision"].mean()
        rec  = cv["test_recall"].mean()
        f1   = cv["test_f1"].mean()

        means[i, j] = mean
        stds[i, j]  = std
        rows.append((mf, md, mean, std, prec, rec, f1))

        if mean > best_mean or (mean == best_mean and f1 > best_f1_gs):
            best_mean, best_f1_gs = mean, f1
            best_mf, best_md      = mf, md
            best_cfg              = (mf, md, mean, std)
            best_fold_scores      = cv["test_accuracy"]

        sep = "\n" if prev_mf and mf != prev_mf else ""
        print(f"{sep}  mf={mf:<7,}  md={md:<5}  {mean*100:>7.2f}%  "
              f"+-{std*100:.2f}%  {prec*100:>6.2f}%  "
              f"{rec*100:>6.2f}%  {f1:.4f}")
        prev_mf = mf

best_mf, best_md, best_cv, best_std = best_cfg
print("=" * 88)
print(f"\n  Best config: max_features={best_mf:,}, min_df={best_md}")
print(f"  CV Acc : {best_cv*100:.2f}%  +-{best_std*100:.2f}%")
print(f"  CV F1  : {best_f1_gs:.4f}")
print(f"  vs Step 4 baseline: {baseline:.2f}%  "
      f"({'↑' if best_cv*100 > baseline else '↓'} "
      f"{abs(best_cv*100-baseline):.2f}pp)")


# =============================================================================
# FIGURE 2 — Grid Search Heatmap (Accuracy + Std)
# =============================================================================
print("\nGenerating Figure 2: Grid search heatmap...")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes2[0].imshow(means * 100, cmap="YlOrRd", aspect="auto")
for i in range(len(max_features_opts)):
    for j in range(len(min_df_opts)):
        star  = " *" if (max_features_opts[i] == best_mf
                         and min_df_opts[j] == best_md) else ""
        color = "white" if means[i,j]*100 > (means.max()*100 - 2) else "black"
        axes2[0].text(j, i, f"{means[i,j]*100:.2f}%{star}",
                      ha="center", va="center", fontsize=9, color=color)
axes2[0].set_xticks(range(len(min_df_opts)))
axes2[0].set_xticklabels([f"min_df={v}" for v in min_df_opts], fontsize=10)
axes2[0].set_yticks(range(len(max_features_opts)))
axes2[0].set_yticklabels([f"{v:,}" for v in max_features_opts], fontsize=10)
axes2[0].set_xlabel("min_df", fontsize=11)
axes2[0].set_ylabel("max_features", fontsize=11)
axes2[0].set_title("CV Mean Accuracy  (* = best)", fontsize=12, fontweight="bold")
plt.colorbar(im1, ax=axes2[0], label="Accuracy (%)")

im2 = axes2[1].imshow(stds * 100, cmap="Blues_r", aspect="auto")
for i in range(len(max_features_opts)):
    for j in range(len(min_df_opts)):
        star  = " *" if (max_features_opts[i] == best_mf
                         and min_df_opts[j] == best_md) else ""
        color = "white" if stds[i,j]*100 < (stds.min()*100 + 0.3) else "black"
        axes2[1].text(j, i, f"+-{stds[i,j]*100:.2f}%{star}",
                      ha="center", va="center", fontsize=9, color=color)
axes2[1].set_xticks(range(len(min_df_opts)))
axes2[1].set_xticklabels([f"min_df={v}" for v in min_df_opts], fontsize=10)
axes2[1].set_yticks(range(len(max_features_opts)))
axes2[1].set_yticklabels([f"{v:,}" for v in max_features_opts], fontsize=10)
axes2[1].set_xlabel("min_df", fontsize=11)
axes2[1].set_ylabel("max_features", fontsize=11)
axes2[1].set_title("CV Std — lower = more stable  (* = best mean)", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes2[1], label="Std (%)")

fig2.suptitle(
    f"TF-IDF Grid Search — Complement NB  (alpha={BEST_ALPHA}, 5-Fold CV, n=490)",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "fig2_gridsearch_heatmap.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved -> {p2}")


# =============================================================================
# FIGURE 3 — Accuracy vs Stability Error Bar (with F1 overlay)
# =============================================================================
print("Generating Figure 3: Accuracy vs stability error bar plot...")

fig3, ax3 = plt.subplots(figsize=(14, 6))

x_labels  = [f"mf={mf//1000}k/md={md}" for mf, md, _, _, _, _, _ in rows]
cv_means_ = [m * 100 for _, _, m, _, _, _, _ in rows]
cv_stds_  = [s * 100 for _, _, _, s, _, _, _ in rows]
cv_f1s_   = [f * 100 for _, _, _, _, _, _, f in rows]
x_pos     = np.arange(len(rows))
md_colors = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 5: "#9C27B0"}
bar_cols  = [md_colors[md] for _, md, _, _, _, _, _ in rows]

bars3 = ax3.bar(x_pos, cv_means_, color=bar_cols, alpha=0.75, width=0.6, zorder=2)
ax3.errorbar(x_pos, cv_means_, yerr=cv_stds_,
             fmt="none", color="black", capsize=4, capthick=1.5,
             linewidth=1.5, zorder=3)

# F1 as line overlay
ax3_f1 = ax3.twinx()
ax3_f1.plot(x_pos, cv_f1s_, "D--", color="crimson", markersize=5,
            linewidth=1.2, alpha=0.7, label="Macro F1 (%)")
ax3_f1.set_ylabel("Macro F1 (%)", fontsize=10, color="crimson")
ax3_f1.tick_params(axis="y", labelcolor="crimson")

best_i = next(k for k, (mf, md, _, _, _, _, _) in enumerate(rows)
              if mf == best_mf and md == best_md)
bars3[best_i].set_edgecolor("red")
bars3[best_i].set_linewidth(2.5)
ax3.annotate(
    f"Best\n{best_cv*100:.2f}% +-{best_std*100:.2f}%",
    xy=(best_i, best_cv*100 + best_std*100 + 0.2),
    ha="center", va="bottom", fontsize=9, color="red", fontweight="bold"
)

ax3.axhline(baseline, color="gray", linestyle="--", linewidth=1.4,
            zorder=1, label=f"Step 4 baseline (mf=10k, md=2): {baseline:.2f}%")
for sep in [3.5, 7.5, 11.5]:
    ax3.axvline(sep, color="lightgray", linewidth=1, linestyle=":")
for gi, mf_val in enumerate(max_features_opts):
    ax3.text(gi*4 + 1.5, min(cv_means_) - 1.8,
             f"max_features={mf_val:,}", ha="center", fontsize=8, color="dimgray")

legend_patches = [mpatches.Patch(color=c, alpha=0.75, label=f"min_df={k}")
                  for k, c in md_colors.items()]
legend_patches.append(
    mpatches.Patch(edgecolor="red", facecolor="none", linewidth=2.5,
                   label="Best config (red border)"))
ax3.legend(handles=legend_patches, loc="lower right", fontsize=9,
           title="min_df", framealpha=0.9)
ax3_f1.legend(loc="upper right", fontsize=9, framealpha=0.7)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, fontsize=7.5, rotation=30, ha="right")
ax3.set_ylabel("5-Fold CV Accuracy (%)", fontsize=11)
ax3.set_title(
    "Accuracy vs. Stability — All TF-IDF Configurations  "
    "(error bars = CV std, red line = Macro F1)",
    fontsize=12, fontweight="bold"
)
ax3.set_ylim(min(cv_means_) - max(cv_stds_) - 2.5,
             max(cv_means_) + max(cv_stds_) + 2.0)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, "fig3_accuracy_errorbar.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved -> {p3}")


# =============================================================================
# FIGURE 4 — Per-Fold Accuracy of Best Configuration
# =============================================================================
print("Generating Figure 4: Per-fold accuracy of best configuration...")

fig4, ax4 = plt.subplots(figsize=(9, 5))
fold_nums = np.arange(1, N_SPLITS + 1)
fold_accs = best_fold_scores * 100
fold_mean = fold_accs.mean()
fold_std  = fold_accs.std()

fold_colors = []
for v in fold_accs:
    if v == fold_accs.max():   fold_colors.append(TEST_COLOR)
    elif v == fold_accs.min(): fold_colors.append(TRAIN_COLOR)
    else:                      fold_colors.append("#5BA4CF")

bars4 = ax4.bar(fold_nums, fold_accs, color=fold_colors,
                alpha=0.82, width=0.55, zorder=2,
                edgecolor="white", linewidth=0.8)
for bar, val in zip(bars4, fold_accs):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.15,
             f"{val:.2f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax4.axhline(fold_mean, color="black", linewidth=2.0, linestyle="-",
            zorder=3, label=f"Mean = {fold_mean:.2f}%")
ax4.axhspan(fold_mean - fold_std, fold_mean + fold_std,
            alpha=0.12, color="black", zorder=1,
            label=f"+-1 Std band (+-{fold_std:.2f}%)")
ax4.set_xticks(fold_nums)
ax4.set_xticklabels([f"Fold {i}" for i in fold_nums], fontsize=11)
ax4.set_ylabel("Accuracy (%)", fontsize=11)
ax4.set_xlabel("Cross-Validation Fold", fontsize=11)
ax4.set_title(
    f"Per-Fold Accuracy — Best Config  "
    f"(max_features={best_mf:,}, min_df={best_md}, alpha={BEST_ALPHA})\n"
    f"Mean={fold_mean:.2f}%  |  Std=+-{fold_std:.2f}%  |  "
    f"Range=[{fold_accs.min():.2f}%, {fold_accs.max():.2f}%]",
    fontsize=11, fontweight="bold"
)
ax4.set_ylim(fold_accs.min() - fold_std - 2.0,
             fold_accs.max() + fold_std + 2.0)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
color_patches = [
    mpatches.Patch(color=TEST_COLOR,  alpha=0.82, label="Highest fold"),
    mpatches.Patch(color="#5BA4CF",   alpha=0.82, label="Middle folds"),
    mpatches.Patch(color=TRAIN_COLOR, alpha=0.82, label="Lowest fold"),
    mpatches.Patch(color="black",     alpha=0.12,
                   label=f"+-1 Std band (+-{fold_std:.2f}%)"),
]
ax4.legend(handles=color_patches, fontsize=9, loc="lower right", framealpha=0.9)
plt.tight_layout()
p4 = os.path.join(FIGURES_DIR, "fig4_best_config_folds.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved -> {p4}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*65}")
print(f"  STEP 4b COMPLETE — ALL FIGURES SAVED")
print(f"{'='*65}")
print(f"  fig1  Stratified K-Fold split    -> fold balance transparency")
print(f"  fig2  Grid search heatmap        -> accuracy + std across all configs")
print(f"  fig3  Accuracy vs stability      -> error bar + F1 overlay vs baseline")
print(f"  fig4  Per-fold best config       -> fold-by-fold stability proof")
print(f"\n  Best config found:")
print(f"    BEST_ALPHA        = {BEST_ALPHA}")
print(f"    BEST_MAX_FEATURES = {best_mf}")
print(f"    BEST_MIN_DF       = {best_md}")
print(f"    CV Acc            = {best_cv*100:.2f}%  +-{best_std*100:.2f}%")
print(f"    CV F1             = {best_f1_gs:.4f}")
print(f"\n-> Open 5final_eval.py and set:")
print(f"     BEST_ALPHA        = {BEST_ALPHA}")
print(f"     BEST_MAX_FEATURES = {best_mf}")
print(f"     BEST_MIN_DF       = {best_md}")
print(f"-> Then run: py 5final_eval.py  (unseals test_210.csv — one time only)")
print(f"-> Test set (test_210.csv) still SEALED.")
