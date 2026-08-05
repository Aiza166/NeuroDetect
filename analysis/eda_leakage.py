"""EDA + target-leakage screen for data/raw/parkinsons.csv.

Reports: shape, class balance, per-feature correlation and mutual information
with the target, and a synthetic-data screen (duplicates, uniformity, cleanliness).

No tuning, no feature selection beyond what is explicitly stated.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

SEED = 42
RAW = "data/raw/parkinsons.csv"
TARGET = "Diagnosis"
ID_COLS = ["PatientID", "DoctorInCharge"]

df = pd.read_csv(RAW)

print("=" * 70)
print("1. SHAPE AND CLASS BALANCE")
print("=" * 70)
print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
feat = df.drop(columns=[c for c in ID_COLS if c in df.columns] + [TARGET])
print(f"Feature matrix: {feat.shape[1]} features (dropped {ID_COLS} + target)")
print(f"Missing values: {int(df.isna().sum().sum())}")

counts = df[TARGET].value_counts().sort_index()
print("\nClass balance:")
for k, v in counts.items():
    print(f"  {TARGET}={k}: {v:5d}  ({v / len(df):6.2%})")
print(f"  majority-class baseline accuracy: {counts.max() / len(df):.4f}")

print()
print("=" * 70)
print("2. FEATURE ASSOCIATION WITH TARGET (ranked)")
print("=" * 70)
y = df[TARGET].values
X = feat

pearson = X.apply(lambda c: np.corrcoef(c, y)[0, 1])
mi = mutual_info_classif(X, y, random_state=SEED)

rank = pd.DataFrame({
    "feature": X.columns,
    "pearson_r": pearson.values,
    "abs_r": np.abs(pearson.values),
    "mutual_info": mi,
}).sort_values("mutual_info", ascending=False).reset_index(drop=True)
rank.index += 1

pd.set_option("display.width", 120)
print(rank[["feature", "pearson_r", "abs_r", "mutual_info"]].to_string(
    float_format=lambda v: f"{v: .4f}"))

print()
print("=" * 70)
print("4. SYNTHETIC-DATA SCREEN")
print("=" * 70)

print(f"\n[a] Exact duplicate rows (all columns):        {df.duplicated().sum()}")
print(f"[a] Exact duplicate rows (features only):     {X.duplicated().sum()}")
print(f"[a] Duplicate PatientIDs:                     {df['PatientID'].duplicated().sum()}")
print(f"[a] PatientID contiguous range?               "
      f"{sorted(df['PatientID']) == list(range(df['PatientID'].min(), df['PatientID'].min() + len(df)))}")

print("\n[b] Uniformity screen for continuous features")
print("    (KS test vs. Uniform(min,max); p > 0.05 => indistinguishable from uniform)")
cont = [c for c in X.columns if X[c].nunique() > 20]
rows = []
for c in cont:
    v = X[c].values
    lo, hi = v.min(), v.max()
    ks = stats.kstest(v, "uniform", args=(lo, hi - lo))
    rows.append({
        "feature": c, "min": lo, "max": hi,
        "mean": v.mean(), "std": v.std(),
        "skew": stats.skew(v), "kurtosis": stats.kurtosis(v),
        "ks_p_vs_uniform": ks.pvalue,
    })
u = pd.DataFrame(rows).sort_values("ks_p_vs_uniform", ascending=False)
print(u.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
n_unif = (u["ks_p_vs_uniform"] > 0.05).sum()
print(f"\n    -> {n_unif}/{len(u)} continuous features are statistically "
      f"indistinguishable from a uniform distribution.")

print("\n[c] Binary/categorical features: observed vs. plausible-clean rates")
disc = [c for c in X.columns if X[c].nunique() <= 20]
for c in disc:
    vc = X[c].value_counts(normalize=True).sort_index()
    print(f"    {c:28s} n_levels={X[c].nunique():2d}  "
          + " ".join(f"{k}:{v:.3f}" for k, v in vc.items()))

print("\n[d] Inter-feature correlation structure")
corr = X.corr().abs()
np.fill_diagonal(corr.values, np.nan)
print(f"    max |r| between any two features: {np.nanmax(corr.values):.4f}")
print(f"    mean |r| across all pairs:        {np.nanmean(corr.values):.4f}")
print("    (real clinical data has correlated blocks, e.g. BMI/BP/cholesterol;")
print("     near-zero everywhere implies independently sampled columns)")

print("\n[e] Clinical-score ranges vs. real instrument ranges")
for c, real in [("UPDRS", "0-199"), ("MoCA", "0-30"), ("FunctionalAssessment", "0-10")]:
    if c in X.columns:
        print(f"    {c:22s} observed [{X[c].min():.3f}, {X[c].max():.3f}]  "
              f"instrument range {real}  n_unique={X[c].nunique()}")
