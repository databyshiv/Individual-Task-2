"""
Individual Task 2, Part 2: bias, fairness and learning-curve analysis.

Reproduces the Random Forest pipeline from Individual Task 1 (Pipeline 2,
Taiwanese bankruptcy prediction) and extends it with:
  (2) a learning curve across varying training-set sizes,
  (3) a Fairlearn disaggregated fairness audit using firm scale as a
      justified proxy sensitive attribute.

Run:  python3 task2_analysis.py
"""
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)

# ---------------------------------------------------------------- settings
DATA_PATH = Path(
    "taiwan_bankruptcy.csv"
)
OUT_DIR = Path("outputs_task2")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42          # identical to Task 1
TEST_SIZE = 0.25           # identical to Task 1
SCALE_PROXY = "Total assets to GNP price"

# Okabe-Ito colourblind-safe palette
C_TRAIN, C_CV = "#0072B2", "#D55E00"
SEQ = ["#BDD7E7", "#6BAED6", "#3182BD", "#08519C"]   # single-hue sequential
INK, MUTED = "#1a1a1a", "#666666"

results = {}


def style(ax):
    """Recessive grid and axes; no chartjunk."""
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------- load
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
target = next(c for c in df.columns if "bankrupt" in c.lower())
y = df[target].astype(int)
X = df.drop(columns=[target])

constant = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=constant)

print(f"Loaded {X.shape[0]:,} companies x {X.shape[1]} features; "
      f"bankrupt rate = {y.mean():.2%}")
print(f"Dropped constant feature(s): {constant}")
results["n_rows"] = int(X.shape[0])
results["n_features"] = int(X.shape[1])
results["bankrupt_rate"] = float(y.mean())
results["dropped_constant"] = constant

# ---------------------------------------------------- reproduce Task 1
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
dummy_acc = dummy.score(X_te, y_te)

rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
y_prob = rf.predict_proba(X_te)[:, 1]

acc = rf.score(X_te, y_te)
roc = roc_auc_score(y_te, y_prob)
pr = average_precision_score(y_te, y_prob)
prec = precision_score(y_te, y_pred)
rec = recall_score(y_te, y_pred)
f1 = 2 * prec * rec / (prec + rec)

cv_ap = cross_val_score(
    rf, X_tr, y_tr, cv=5, scoring="average_precision", n_jobs=-1
)

cv_ap_shuffled = cross_val_score(
    rf, X_tr, y_tr,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="average_precision", n_jobs=-1,
)
lag1 = float(np.corrcoef(y.values[:-1], y.values[1:])[0, 1])
print(f"5-fold CV PR-AUC, shuffled  {cv_ap_shuffled.mean():.3f} "
      f"+/- {cv_ap_shuffled.std():.3f}")
print(f"Target lag-1 autocorrelation in file order: {lag1:+.3f}")
results["cv_diagnostic"] = dict(
    unshuffled_mean=float(cv_ap.mean()), unshuffled_std=float(cv_ap.std()),
    shuffled_mean=float(cv_ap_shuffled.mean()),
    shuffled_std=float(cv_ap_shuffled.std()),
    shuffled_folds=[float(v) for v in cv_ap_shuffled],
    target_lag1_autocorrelation=lag1,
)

print("\n=== REPRODUCTION CHECK vs Task 1 ===")
print(f"Dummy accuracy      {dummy_acc:.4f}   (Task 1: 0.9677)")
print(f"RF accuracy         {acc:.4f}   (Task 1: 0.9620)")
print(f"ROC-AUC             {roc:.3f}    (Task 1: 0.946)")
print(f"PR-AUC              {pr:.3f}    (Task 1: 0.470)")
print(f"Precision(bankrupt) {prec:.3f}    (Task 1: 0.417)")
print(f"Recall(bankrupt)    {rec:.3f}    (Task 1: 0.455)")
print(f"F1(bankrupt)        {f1:.3f}    (Task 1: 0.435)")
print(f"5-fold CV PR-AUC    {cv_ap.mean():.3f} +/- {cv_ap.std():.3f}"
      f"   (Task 1: 0.392 +/- 0.099)")
print(classification_report(y_te, y_pred, digits=3))
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

results["reproduction"] = dict(
    dummy_acc=float(dummy_acc), acc=float(acc), roc_auc=float(roc),
    pr_auc=float(pr), precision=float(prec), recall=float(rec), f1=float(f1),
    cv_ap_mean=float(cv_ap.mean()), cv_ap_std=float(cv_ap.std()),
    cv_folds=[float(v) for v in cv_ap],
    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
)

# ------------------------------------------------- (2) learning curve
print("\n=== LEARNING CURVE ===")
train_sizes, train_scores, cv_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=400, class_weight="balanced", min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    X_tr, y_tr,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="average_precision",
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)
tr_m, tr_s = train_scores.mean(1), train_scores.std(1)
cv_m, cv_s = cv_scores.mean(1), cv_scores.std(1)

for n, a, b in zip(train_sizes, tr_m, cv_m):
    print(f"  n={int(n):5d}  train PR-AUC={a:.3f}  cv PR-AUC={b:.3f}  gap={a-b:.3f}")

results["learning_curve"] = dict(
    train_sizes=[int(v) for v in train_sizes],
    train_mean=[float(v) for v in tr_m], train_std=[float(v) for v in tr_s],
    cv_mean=[float(v) for v in cv_m], cv_std=[float(v) for v in cv_s],
)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
style(ax)
ax.fill_between(train_sizes, tr_m - tr_s, tr_m + tr_s, color=C_TRAIN,
                alpha=0.15, lw=0, zorder=2)
ax.fill_between(train_sizes, cv_m - cv_s, cv_m + cv_s, color=C_CV,
                alpha=0.15, lw=0, zorder=2)
ax.plot(train_sizes, tr_m, color=C_TRAIN, lw=2, marker="o", ms=5,
        label="Training score", zorder=3)
ax.plot(train_sizes, cv_m, color=C_CV, lw=2, marker="s", ms=5,
        label="5-fold cross-validation score", zorder=3)
ax.set_xlabel("Training examples", fontsize=10, color=INK)
ax.set_ylabel("PR-AUC (average precision)", fontsize=10, color=INK)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.annotate(f"{tr_m[-1]:.2f}", (train_sizes[-1], tr_m[-1]), color=C_TRAIN,
            fontsize=9, fontweight="bold", xytext=(-6, 8),
            textcoords="offset points", ha="right")
ax.annotate(f"{cv_m[-1]:.2f}", (train_sizes[-1], cv_m[-1]), color=C_CV,
            fontsize=9, fontweight="bold", xytext=(-6, -16),
            textcoords="offset points", ha="right")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_learning_curve.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ (3) fairness audit
print("\n=== FAIRNESS AUDIT (Fairlearn) ===")
# Quartiles of the scale proxy, computed on the TRAINING split only so the
# bucket boundaries are not informed by test data.
edges = np.quantile(X_tr[SCALE_PROXY], [0.25, 0.50, 0.75])
labels = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
sf_te = pd.cut(X_te[SCALE_PROXY], bins=[-np.inf, *edges, np.inf],
               labels=labels)
print(f"Proxy: {SCALE_PROXY}; train-derived quartile edges = "
      f"{[float(e) for e in edges]}")

metrics = {
    "count": count,
    "base rate": lambda yt, yp: float(np.mean(yt)),
    "selection rate": selection_rate,
    "recall": recall_score,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "FNR": false_negative_rate,
    "FPR": false_positive_rate,
}
mf = MetricFrame(metrics=metrics, y_true=y_te, y_pred=y_pred,
                 sensitive_features=sf_te)
by_group = mf.by_group
print(by_group.to_string())

dpd = demographic_parity_difference(y_te, y_pred, sensitive_features=sf_te)
eod = equalized_odds_difference(y_te, y_pred, sensitive_features=sf_te)
rec_gap = by_group["recall"].max() - by_group["recall"].min()
fnr_gap = by_group["FNR"].max() - by_group["FNR"].min()

print(f"\nOverall recall                   {mf.overall['recall']:.3f}")
print(f"Demographic parity difference    {dpd:.4f}")
print(f"Equalized odds difference        {eod:.4f}")
print(f"Recall gap (max-min)             {rec_gap:.4f}")
print(f"FNR gap (max-min)                {fnr_gap:.4f}")

# positives per group, so small-cell caveats can be stated honestly
pos_per_group = pd.Series(y_te.values, index=sf_te.values).groupby(level=0).sum()
print("\nBankrupt firms in test set per group:")
print(pos_per_group.to_string())

def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion (group recall)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


pos_map = {str(k): int(v) for k, v in pos_per_group.items()}
ci_rows = {}
print("\nGroup recall with 95% Wilson confidence intervals:")
for g in labels:
    n_pos = pos_map.get(g, 0)
    rec_g = float(by_group["recall"].get(g, float("nan")))
    n_tp = int(round(rec_g * n_pos))
    lo, hi = wilson(n_tp, n_pos)
    ci_rows[g] = dict(positives=n_pos, true_positives=n_tp,
                      recall=rec_g, ci_low=lo, ci_high=hi)
    print(f"  {g:13s} {n_tp:2d}/{n_pos:2d}  recall={rec_g:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")

lo_max = max(v["ci_low"] for v in ci_rows.values())
hi_min = min(v["ci_high"] for v in ci_rows.values())
if lo_max < hi_min:
    print(f"  -> all intervals overlap in [{lo_max:.3f}, {hi_min:.3f}]: "
          "no pair of groups separated at 95%.")
    
results["fairness"] = dict(
    proxy=SCALE_PROXY,
    quartile_edges=[float(e) for e in edges],
    by_group=json.loads(by_group.to_json(orient="index")),
    overall={k: float(v) for k, v in mf.overall.items()},
    demographic_parity_difference=float(dpd),
    equalized_odds_difference=float(eod),
    recall_gap=float(rec_gap),
    fnr_gap=float(fnr_gap),
    recall_wilson_ci=ci_rows,
    all_intervals_overlap=bool(lo_max < hi_min),
    positives_per_group={str(k): int(v) for k, v in pos_per_group.items()},
)

# Figure: recall and base rate by firm-scale quartile.
# Two measures on one axis (both are rates in [0,1]) - no dual axis.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
style(ax)
xs = np.arange(len(labels))
w = 0.38
r_vals = by_group["recall"].reindex(labels).values
b_vals = by_group["base rate"].reindex(labels).values
ax.bar(xs - w / 2 - 0.01, r_vals, w, color="#08519C", zorder=3,
       label="Recall on bankrupt firms")
ax.bar(xs + w / 2 + 0.01, b_vals, w, color="#BDD7E7", zorder=3,
       label="Actual bankruptcy base rate")
for x, v in zip(xs - w / 2 - 0.01, r_vals):
    ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=9,
            color=INK, fontweight="bold")
for x, v in zip(xs + w / 2 + 0.01, b_vals):
    ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_xlabel("Firm scale quartile (total assets to GNP price)",
              fontsize=10, color=INK)
ax.set_ylabel("Rate", fontsize=10, color=INK)
ax.set_ylim(0, max(r_vals.max(), b_vals.max()) * 1.28)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_fairness_by_scale.png", dpi=200)
plt.close(fig)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote figures and results.json to {OUT_DIR.resolve()}")
