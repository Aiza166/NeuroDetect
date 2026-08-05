# Target Leakage & Data Provenance Audit — NeuroDetect

**Date:** 2026-08-06
**Dataset audited:** `data/raw/parkinsons.csv` (vendored, 2,105 × 35)
**Question asked:** are UPDRS / MoCA / Functional Assessment leaking the target?
**Answer:** yes — and the problem is worse than that framing suggests.

Everything below is reproducible from two committed scripts:

```bash
python analysis/eda_leakage.py
```

```bash
python analysis/leakage_experiment.py
```

---

## 0. Environment setup, and two things that were wrong before I started

### 0.1 The repo was not present locally

`C:\proj` was empty — no git repo, no data. I cloned both repos you named:

```bash
git clone https://github.com/Aiza166/NeuroDetect.git
git clone https://github.com/Aiza166/neurodetect-clarity.git
```

All analysis below runs against the freshly cloned `NeuroDetect/`.

### 0.2 The target column is `Diagnosis`, not `Parkinsons`

You described the label as `Parkinsons`. The actual CSV header ends:

```text
... Constipation,Diagnosis,DoctorInCharge
```

There is no `Parkinsons` column. `src/models/train_model.py` and
`src/data_pipeline/preprocess_data.py` both correctly use `Diagnosis`; only the
README (and two stray references in `disease-predictor/Parkinson_Detection.ipynb`)
called it `Parkinsons`. I used `Diagnosis` throughout and corrected the README.

### 0.3 TensorFlow was not installed

The repo's model is a Keras `Sequential`. The local env (Python 3.13.5) had
pandas 2.3.0 and scikit-learn 1.7.0 but no TensorFlow, so I installed
`tensorflow-cpu` (resolved to **2.21.0**) rather than substitute a different
model family. This matters: had I fallen back to `sklearn.MLPClassifier`, the
numbers would not be comparable to anything the repo produces. The script still
carries an `sklearn-mlp` fallback path so it runs in a TF-less environment, and
it prints which backend was used (`backend: keras`) so the provenance of any
result is unambiguous.

> Note: TF ≥ 2.11 has no GPU support on native Windows, so this ran on CPU.
> Irrelevant for a 2,105-row dataset.

---

## 1. Dataset shape and class balance

| Property | Value |
|---|---|
| Raw shape | 2,105 rows × 35 columns |
| Dropped as non-features | `PatientID`, `DoctorInCharge` |
| Feature count | 32 |
| Target | `Diagnosis` |
| Missing values | **0** |
| Class 0 (no Parkinson's) | 801 (38.05%) |
| Class 1 (Parkinson's) | 1,304 (61.95%) |
| **Majority-class baseline accuracy** | **0.6195** |

**Why the baseline matters.** The classes are imbalanced roughly 62/38 in favour
of the *positive* class. A model that unconditionally outputs "Parkinson's"
scores 0.6195 accuracy while learning nothing. Any accuracy figure in this
report must be read against 0.6195, not against 0.50. This single number is what
makes Model B's result interpretable at all.

---

## 2. Feature association with the target

### How

Two complementary measures, because each misses something the other catches:

- **Pearson correlation** (`np.corrcoef`) — captures *linear, monotonic*
  association and gives a signed direction. Blind to non-monotonic
  relationships.
- **Mutual information** (`sklearn.feature_selection.mutual_info_classif`,
  `random_state=42`) — captures *any* statistical dependence including
  non-monotonic ones, and handles the mix of continuous and binary columns.
  Unsigned, so it says "how much" but not "which way".

Ranked by mutual information, since that is the more general criterion.
Computed on the **full dataset** — this is descriptive screening for leakage,
not model selection, so there is no split to respect here.

### Results (full 32-feature ranking)

| # | Feature | Pearson r | \|r\| | Mutual info |
|---|---|---|---|---|
| 1 | **UPDRS** | +0.3980 | 0.3980 | **0.1453** |
| 2 | **FunctionalAssessment** | −0.2250 | 0.2250 | **0.0591** |
| 3 | Rigidity | +0.1856 | 0.1856 | 0.0437 |
| 4 | **MoCA** | −0.1731 | 0.1731 | **0.0344** |
| 5 | Tremor | +0.2744 | 0.2744 | 0.0295 |
| 6 | Constipation | +0.0253 | 0.0253 | 0.0119 |
| 7 | Ethnicity | −0.0051 | 0.0051 | 0.0114 |
| 8 | Bradykinesia | +0.1840 | 0.1840 | 0.0097 |
| 9 | SpeechProblems | −0.0122 | 0.0122 | 0.0096 |
| 10 | Stroke | +0.0281 | 0.0281 | 0.0087 |
| 11 | PosturalInstability | +0.1475 | 0.1475 | 0.0079 |
| 12 | BMI | +0.0301 | 0.0301 | 0.0047 |
| 13 | DietQuality | −0.0230 | 0.0230 | 0.0038 |
| 14 | EducationLevel | +0.0046 | 0.0046 | 0.0031 |
| 15 | Age | +0.0653 | 0.0653 | 0.0000 |
| 16 | Smoking | +0.0052 | 0.0052 | 0.0000 |
| 17 | Depression | +0.0591 | 0.0591 | 0.0000 |
| 18 | Diabetes | +0.0571 | 0.0571 | 0.0000 |
| 19 | Hypertension | −0.0116 | 0.0116 | 0.0000 |
| 20 | TraumaticBrainInjury | +0.0230 | 0.0230 | 0.0000 |
| 21 | SleepQuality | −0.0433 | 0.0433 | 0.0000 |
| 22 | FamilyHistoryParkinsons | +0.0134 | 0.0134 | 0.0000 |
| 23 | AlcoholConsumption | +0.0367 | 0.0367 | 0.0000 |
| 24 | PhysicalActivity | +0.0129 | 0.0129 | 0.0000 |
| 25 | Gender | +0.0168 | 0.0168 | 0.0000 |
| 26 | CholesterolTriglycerides | +0.0156 | 0.0156 | 0.0000 |
| 27 | CholesterolHDL | −0.0196 | 0.0196 | 0.0000 |
| 28 | CholesterolLDL | +0.0147 | 0.0147 | 0.0000 |
| 29 | DiastolicBP | −0.0291 | 0.0291 | 0.0000 |
| 30 | SystolicBP | −0.0044 | 0.0044 | 0.0000 |
| 31 | CholesterolTotal | −0.0190 | 0.0190 | 0.0000 |
| 32 | SleepDisorders | −0.0106 | 0.0106 | 0.0000 |

### What this says

1. **The three clinical instruments occupy ranks 1, 2, and 4.** UPDRS alone has
   2.5× the mutual information of the next-best feature.
2. **Ranks 3, 5, 8, 9, 11 are motor signs** — Rigidity, Tremor, Bradykinesia,
   SpeechProblems, PosturalInstability. So the entire top 11 by MI is *clinical
   observations of Parkinson's disease*, in some form.
3. **17 of 32 features have mutual information of exactly 0.0000.** Not "small" —
   the estimator returns a clean zero. This includes every feature you would
   expect to carry genuine epidemiological risk signal: Age, Gender,
   FamilyHistoryParkinsons, all four cholesterol measures, both blood pressures,
   Smoking, PhysicalActivity. In real Parkinson's epidemiology, age is the
   single strongest risk factor by a wide margin. Here it is zero. That was the
   first strong hint the data is generated (see §4).
4. Note the sign disagreement at rank 5 vs 8: Tremor has higher |r| (0.2744)
   than Rigidity (0.1856) but *lower* MI. This is a discretisation artefact of
   the MI estimator on binary features and not meaningful — it is why I did not
   rank on a single metric.

---

## 3. The controlled experiment

### Design, and what "identical" means here

The whole value of this comparison rests on only *one* thing differing between
arms. Held constant:

| Held constant | Value / mechanism |
|---|---|
| Seed | `SEED = 42` — used for the split, `tf.keras.utils.set_random_seed`, and MI |
| Split | **One** `train_test_split` on row *indices*, computed once, reused by every arm |
| Stratification | `stratify=y` — so both arms see the same 62.0% positive test set |
| Split size | 80/20 → train n=1,684, test n=421 |
| Architecture | `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` → `Dropout(0.2)` → `Dense(1, sigmoid)` |
| Optimiser / loss | `adam` / `binary_crossentropy` |
| Training | max 50 epochs, batch 32, `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, `validation_split=0.2` |
| Decision threshold | **0.50**, unmodified |
| Determinism | `TF_DETERMINISTIC_OPS=1`, `PYTHONHASHSEED=42` |

The **only** difference between arms is which columns are in the matrix.

### Two deliberate methodology choices

**(a) I split on indices, not on data.** A subtle trap: calling
`train_test_split(X_a, y)` and `train_test_split(X_b, y)` with the same seed
does give the same partition, but it is fragile and easy to get wrong. Splitting
`np.arange(len(df))` once and slicing both feature matrices with those indices
makes it *structurally impossible* for the two arms to see different patients.

**(b) I fit the scaler on the training fold only — the repo does not.**
`src/data_pipeline/preprocess_data.py` does this:

```python
X_scaled = scaler.fit_transform(X)   # BEFORE any train/test split
```

`fit_transform` on the whole frame computes the mean and standard deviation from
*all* rows, test rows included, and bakes them into the training features. That
is a second, independent leak — a preprocessing leak — layered underneath the
one you asked about. It is mild for standardisation on 2,105 rows, but it means
the repo's own reported test accuracy is not a clean held-out estimate. My
script fits the scaler on `idx_train` only and applies it to test. I did **not**
change `preprocess_data.py` itself, since that was outside what you asked for —
but it should be fixed.

### Arms

- **Model A** — all 32 features.
- **Model B** — 29 features; `UPDRS`, `MoCA`, `FunctionalAssessment` removed, exactly as you specified.
- **Model C** *(supplementary, my addition)* — 24 features; also removes
  `Tremor`, `Rigidity`, `Bradykinesia`, `PosturalInstability`, `SpeechProblems`.

I added Model C because §2 made clear that your three named columns are not the
only leaky ones. Motor signs are *cardinal diagnostic criteria* for Parkinson's —
bradykinesia plus rigidity or tremor is essentially the clinical definition. A
model using them is not predicting a diagnosis, it is restating one. Model B on
its own would have understated the problem, so answering your question honestly
required the third arm. It is labelled supplementary everywhere and does not
replace the A/B comparison you asked for.

### No tuning — per your instruction

Every hyperparameter above was fixed before the first run and taken from the
repo's existing `train_model.py`. I ran each arm **once**. I did not adjust
layer sizes, learning rate, epochs, class weights, or the decision threshold;
I did not try alternate seeds and report the best; I did not touch Model B after
seeing it score poorly. Model C's near-random result is reported as-is.

---

## 4. Results

Test set: n = 421, 62.00% positive. **Baseline to beat: 0.6200.**

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **A** — all features | 32 | 0.8504 | 0.8694 | 0.8927 | 0.8809 | 0.9121 |
| **B** — clinical scores removed | 29 | 0.6912 | 0.7148 | 0.8352 | 0.7703 | 0.7368 |
| *C* — + motor signs removed *(suppl.)* | 24 | *0.6081* | *0.6171* | *0.9693* | *0.7541* | *0.5302* |

**A → B delta:** accuracy −0.1591 · precision −0.1546 · recall −0.0575 ·
F1 −0.1106 · **ROC-AUC −0.1753**

### Confusion matrices (rows = actual, columns = predicted)

```text
Model A (32 features)            Model B (29 features)            Model C (24 features, suppl.)
         pred 0   pred 1                  pred 0   pred 1                  pred 0   pred 1
act 0       125       35         act 0        73       87         act 0         3      157
act 1        28      233         act 1        43      218         act 1         8      253
```

### Reading the confusion matrices — this is where the story is

The matrices say more than the scalar metrics do.

- **Model A** is a genuinely functional classifier. It identifies 125 of 160
  negatives (78.1% specificity) and 233 of 261 positives.
- **Model B** collapses on the negative class: specificity drops from 78.1% to
  **45.6%** (73/160). It is now wrong more often than right on healthy patients.
  Recall barely moves (0.8927 → 0.8352) because the model compensates by
  predicting "positive" more liberally — which the majority-positive class
  balance rewards.
- **Model C** is the tell. It predicts positive for **410 of 421** test patients.
  It finds 3 true negatives out of 160. Its recall of **0.9693 is the highest of
  all three models** — and it is completely worthless, because it is achieved by
  saying yes to nearly everyone. Its F1 of 0.7541 is within 0.016 of Model B's
  0.7703, which is exactly why F1 alone would have hidden this. ROC-AUC is the
  metric that exposes it: **0.5302**, where 0.5000 is a coin flip.

This is the clearest possible demonstration of why the majority baseline and the
confusion matrix both had to be reported. Accuracy 0.6081 and F1 0.7541 look
like a mediocre-but-working model. They are not. They are the arithmetic of a
constant classifier.

### The verdict on your question

**Your leakage concern is correct, and understated.**

Three of 32 columns carry most of the model's discriminative power. Removing
them costs 17.5 ROC-AUC points and leaves accuracy at 0.6912 against a 0.6200
baseline — about 7 points of real signal.

But the deeper finding is Model C: once *every* feature that is an observation of
Parkinson's disease is removed, **predictive performance is indistinguishable
from random (AUC 0.5302)**. The remaining 24 features — age, sex, BMI, blood
pressure, cholesterol, lifestyle, family history, comorbidities — contain
essentially **zero** information about the label.

So Model A's 0.9121 AUC is not a model that detects Parkinson's from risk
factors. It is a model that reads clinical assessments and motor exam findings
which a clinician recorded *because* they had already diagnosed the patient. The
arrow of inference runs backwards. This is textbook target leakage, and §5
explains why it is baked into the data by construction.

---

## 5. Is the dataset synthetic? Yes, unambiguously

### How I tested

Five independent screens, all in `analysis/eda_leakage.py`.

#### (a) Duplicate rows — the check that came back *too* clean

| Check | Result |
|---|---|
| Exact duplicate rows (all columns) | **0** |
| Exact duplicate rows (features only) | **0** |
| Duplicate `PatientID` | **0** |
| `PatientID` a perfectly contiguous integer range | **True** |

You asked me to look for duplicates as evidence of synthesis. The result is the
*opposite* of what you might expect, and it is more damning. Zero duplicates
across 2,105 rows and 32 features — combined with zero missing values — is not
what clean data looks like. It is what *generated* data looks like. Real clinical
registries have duplicate submissions, re-admissions, missing labs, and
out-of-range typos. A perfectly contiguous `PatientID` range with no gaps means
no record was ever deleted, merged, or excluded.

#### (b) Uniformity — Kolmogorov–Smirnov test against `Uniform(min, max)`

For each of the 15 continuous features (>20 unique values), a one-sample KS test
against a uniform distribution fitted to that column's own observed range.
`p > 0.05` means we cannot distinguish the column from uniform noise.

| Feature | min | max | mean | std | skew | **excess kurtosis** | **KS p** |
|---|---|---|---|---|---|---|---|
| SleepQuality | 4.00 | 10.00 | 7.00 | 1.75 | +0.009 | −1.226 | **0.783** |
| PhysicalActivity | 0.00 | 10.00 | 5.02 | 2.89 | +0.026 | −1.234 | **0.771** |
| FunctionalAssessment | 0.00 | 9.99 | 4.99 | 2.93 | +0.011 | −1.212 | **0.564** |
| CholesterolTriglycerides | 50.11 | 400.0 | 222.9 | 101.9 | +0.017 | −1.206 | **0.562** |
| CholesterolHDL | 20.03 | 99.98 | 59.67 | 23.37 | +0.002 | −1.180 | **0.538** |
| CholesterolLDL | 50.02 | 199.99 | 126.1 | 43.40 | −0.018 | −1.205 | **0.489** |
| MoCA | 0.02 | 29.97 | 15.09 | 8.64 | −0.024 | −1.223 | **0.473** |
| AlcoholConsumption | 0.00 | 19.99 | 10.04 | 5.69 | −0.019 | −1.184 | **0.408** |
| DietQuality | 0.00 | 10.00 | 4.91 | 2.87 | +0.035 | −1.177 | **0.348** |
| CholesterolTotal | 150.1 | 299.96 | 226.9 | 43.58 | −0.051 | −1.199 | **0.152** |
| BMI | 15.01 | 40.00 | 27.21 | 7.21 | +0.035 | −1.219 | **0.084** |
| UPDRS | 0.03 | 198.95 | 101.4 | 56.58 | −0.048 | −1.157 | **0.071** |
| DiastolicBP | 60.0 | 119.0 | 90.25 | 17.06 | −0.057 | −1.173 | 0.024 |
| Age | 50.0 | 89.0 | 69.60 | 11.59 | −0.033 | −1.184 | 0.014 |
| SystolicBP | 90.0 | 179.0 | 133.7 | 26.50 | +0.029 | −1.244 | 0.009 |

**12 of 15 are statistically indistinguishable from uniform.** And the three
that "fail" (Age, SystolicBP, DiastolicBP) fail only because they are
*integer-rounded* uniforms — the KS test detects the discreteness, not any
departure in shape. Their kurtosis is right there with the others.

**The kurtosis column is the smoking gun.** The excess kurtosis of a continuous
uniform distribution is exactly **−1.2**. Every single feature lands between
−1.157 and −1.244, and every skew is within ±0.06 of zero. Real biomarkers are
not uniform. BMI is right-skewed with a long upper tail. Triglycerides are
strongly right-skewed — famously log-normal. Cholesterol is roughly normal.
None of them are flat. Getting −1.2 kurtosis on all fifteen columns
simultaneously happens when someone calls a uniform random number generator
fifteen times.

Note the ranges too: `[15.0, 40.0]` for BMI, `[50, 400]` for triglycerides,
`[150, 300]` for total cholesterol, `[50, 89]` for age. These are round,
human-chosen bounds — the parameters of a generator, not the extremes of a
sampled population.

#### (c) Inter-feature correlation structure

| Statistic | Value |
|---|---|
| max \|r\| between any two of 32 features | **0.0758** |
| mean \|r\| across all pairs | **0.0180** |

This is decisive on its own. In real clinical data, `CholesterolTotal` is
*arithmetically* tied to `CholesterolLDL + CholesterolHDL +
CholesterolTriglycerides/5` (the Friedewald relationship) — you would expect
|r| well above 0.5 within that block. `SystolicBP` and `DiastolicBP` typically
correlate at r ≈ 0.7. BMI correlates with both BP and lipids. Age correlates
with nearly everything.

Here the maximum correlation *anywhere in the entire 32×32 matrix* is 0.076,
which is roughly the sampling noise floor at n=2,105. **Every column was drawn
independently.** No generative process linked them. This also explains the 17
zero-MI features in §2: they are independent noise with respect to the label,
because that is literally how they were made.

#### (d) Clinical score granularity — the most specific proof

| Feature | Observed range | Real instrument range | Unique values / 2,105 rows |
|---|---|---|---|
| UPDRS | [0.028, 198.954] | 0–199 | **2,105** |
| MoCA | [0.021, 29.970] | 0–30 | **2,105** |
| FunctionalAssessment | [0.002, 9.993] | 0–10 | **2,105** |

Every single row has a distinct floating-point value. Consider what these
instruments actually are:

- **MoCA** (Montreal Cognitive Assessment) is a 30-point test scored by summing
  integer points for discrete completed tasks. A MoCA score is an **integer from
  0 to 30**. There are 31 possible values. The dataset contains a value of
  `29.181289291248678`. **This is not a possible MoCA score.**
- **UPDRS** is likewise a sum of integer-scored clinical items. A real UPDRS
  total is an integer.

A continuous float uniformly filling `[0, 30]` is not a cognitive assessment
that was administered to a patient. It is `uniform(0, 30)`. The same holds for
UPDRS filling `[0, 199]` and FunctionalAssessment filling `[0, 10]` — each one
spanning its nominal range end to end, mean sitting almost exactly at the
midpoint (15.09 vs 15.0; 101.4 vs 99.5; 4.99 vs 5.0).

#### (e) Categorical base rates

The binary flags sit at suspiciously round, plausible-looking rates: Gender
0.507/0.493, Stroke 0.049, Hypertension 0.146, Diabetes 0.148,
FamilyHistoryParkinsons 0.146. These are individually credible — which is the
point. Whoever generated this chose sensible-looking marginal probabilities.
What they did not do is generate any *joint* structure between the columns,
which is what (c) detects.

### Provenance corroboration

`README.md` cited `kaggle.com/datasets/nidaguler/parkinsons-dataset` → **HTTP 404**.
The dataset is gone; the vendored CSV in `data/raw/` is the only copy.

But `src/data_pipeline/download_data.py` requests a **different** slug:

```python
kagglehub.dataset_download("rabieelkharoua/parkinsons-disease-dataset-analysis")
```

→ **HTTP 200, still live.** That publisher distributes datasets that are
explicitly labelled synthetic/generated for educational use. The README and the
download script disagree about where this data came from, and the live one is a
known synthetic source. This independently corroborates every statistical
finding above.

### Conclusion on synthesis

**Certain.** Uniform marginals with −1.2 kurtosis across all 15 continuous
features, zero inter-column correlation, non-integer values for integer-scored
clinical instruments, zero duplicates, zero missing values, contiguous IDs, and a
synthetic-data publisher as the actual source. Any one of these would raise
suspicion; together they are conclusive.

**What this means for the leakage question.** The two findings are the same
finding. The generator appears to have drawn most columns as independent noise
and then injected a relationship between the label and a handful of
Parkinson's-specific columns. That is precisely why Model A works, Model B
degrades, and Model C is random. The "leakage" is not an accident of feature
selection — it is the only structure the dataset has.

---

## 6. What I changed

| Path | Change |
|---|---|
| `analysis/eda_leakage.py` | **New.** Shape, class balance, correlation + MI ranking, synthetic screens (a)–(e). |
| `analysis/leakage_experiment.py` | **New.** Three-arm controlled experiment; writes `analysis/results.json`. |
| `analysis/results.json` | **New.** Machine-readable metrics for all arms. |
| `README.md` | Corrected target name to `Diagnosis`; added shape/balance/baseline; added dead-source + vendored-CSV note and the slug discrepancy; added synthetic-data warning table; added metrics table, MI ranking, confusion matrices, interpretation. |
| `ANALYSIS.md` | **New.** This document. |

Not changed, deliberately: `preprocess_data.py`, `train_model.py`,
`predict.py`, the notebook, and the saved `models/*` artifacts. The
preprocessing leak in §3(b) is real and worth fixing, but rewriting the
training pipeline was outside what you asked for.

---

## 7. Recommendations

1. **Fix the preprocessing leak.** Move the `StandardScaler` fit inside the
   training fold in `preprocess_data.py`, or better, into an
   `sklearn.Pipeline` so it cannot be got wrong. The repo's currently reported
   accuracy is not a clean held-out number.
2. **Do not ship Model A's 0.9121 AUC as a headline.** It is a leakage artefact.
   If the frontend collects UPDRS/MoCA/FunctionalAssessment as user inputs — and
   `neurodetect-clarity`'s `plan.md` shows a "Clinical Scores" form section that
   does exactly that — then the app is asking users to supply the answer and
   returning it to them as a prediction.
3. **Relabel the project honestly.** This is a working end-to-end ML pipeline
   demonstrated on synthetic data. That is a perfectly respectable thing for it
   to be. It is not a Parkinson's screening tool, and the README should not imply
   clinical utility. The disclaimer in the frontend plan ("For research purposes
   only") is good but does not go far enough — the data is not real.
4. **If you want a defensible modelling result**, the target has to be
   predictable from features that *precede* diagnosis. On this dataset that is
   impossible (Model C, AUC 0.5302). It requires different data — e.g. the UCI
   Parkinson's voice dataset, where acoustic features are genuine measurements
   taken independently of the clinical assessment.
5. **Keep the vendored CSV under version control** and note its checksum. Since
   the cited source is dead, `data/raw/parkinsons.csv` is now the primary record.

---

## Appendix — the `neurodetect-clarity` repo

You said you'd forgotten why this repo existed. From its history and contents:

It is the **frontend for this project**, generated with
[Lovable](https://lovable.dev) (note the `.lovable/plan.md` file and the
"Changes" / "template: tanstack_start_ts" commit messages). It is a
TanStack-Start + TypeScript + Tailwind + shadcn/ui app.

The reason it is separate: **NeuroDetect commit `5023991` is
"Delete neurodetect-frontend directory"**. You removed the in-repo frontend from
`NeuroDetect` and rebuilt it as this standalone Lovable project. That is the
missing link.

Its `.lovable/plan.md` describes a three-page UI — landing (`/`), prediction form
(`/predict`), result (`/result`) — with the form split into Demographics, Health
Metrics, and **Clinical Scores (UPDRS 0–199, MoCA 0–30, Functional Assessment
0–10)**. It is explicitly a UI skeleton: "No backend, no ML logic."

Two things worth flagging:

- The form asks the user to enter UPDRS, MoCA, and Functional Assessment. Per
  §4, those three inputs are where nearly all of the model's apparent accuracy
  comes from. A user who has those three numbers has already been clinically
  assessed. This is recommendation 2 above, made concrete.
- The MoCA and UPDRS fields are specified as `number` inputs over the correct
  integer instrument ranges — which is more faithful to the real instruments
  than the training data is (§5d). Real integer inputs fed to a model trained on
  uniform floats is its own mismatch.
