# 🧠 NeuroDetect – Parkinson’s Disease Detection

Predicting the likelihood of Parkinson’s Disease using clinical and demographic data with deep learning.


## 📁 Dataset

- **Target**: `Diagnosis` (binary: 0 or 1) — 2,105 rows × 35 columns, no missing values
- **Class balance**: 1,304 positive (61.95%) / 801 negative (38.05%) → majority-class baseline accuracy **0.6195**
- **Features**: 32, after dropping `PatientID` and `DoctorInCharge`. Age, Gender, Sleep Quality, BMI, Family History, UPDRS, MoCA, Functional Assessment, and more.

### ⚠️ Source availability

The Kaggle dataset originally cited in this README
(`nidaguler/parkinsons-dataset`) **is no longer available — the URL returns
HTTP 404.** The raw CSV is therefore **vendored in-repo at
[`data/raw/parkinsons.csv`](data/raw/parkinsons.csv)**, which is the
authoritative copy for reproducing everything below. Do not expect
`src/data_pipeline/download_data.py` to reproduce it.

Note that `download_data.py` actually references a *different* slug,
`rabieelkharoua/parkinsons-disease-dataset-analysis`, which is still live.
That publisher distributes **explicitly synthetic** datasets, which is
consistent with the synthetic-data findings below.

### ⚠️ This dataset is synthetic

Statistical screening ([`analysis/eda_leakage.py`](analysis/eda_leakage.py))
shows this is generated data, not real patient records:

| Check | Finding |
|---|---|
| Uniform distributions | **12 of 15** continuous features are statistically indistinguishable from `Uniform(min, max)` (KS test, p > 0.05) |
| Excess kurtosis | **every** continuous feature sits at ≈ −1.18 to −1.24; the exact value for a uniform distribution is −1.2 |
| Inter-feature correlation | max \|r\| between any two features = **0.076**, mean \|r\| = **0.018** — columns were sampled independently. Real clinical data has correlated blocks (BMI↔BP↔cholesterol) |
| Clinical score granularity | `UPDRS`, `MoCA`, `FunctionalAssessment` each have **2,105 unique float values across 2,105 rows**. Real UPDRS and MoCA are integer-scored instruments and cannot take values like `29.181289291248678` |
| Range fit | UPDRS spans `[0.03, 198.95]`, MoCA `[0.02, 29.97]` — each fills its nominal instrument range almost exactly end to end |
| Exact duplicate rows | 0 (features-only: 0); `PatientID` is a perfectly contiguous integer range |

Zero duplicates plus zero missing values plus uniform marginals plus
uncorrelated columns is the signature of a generator, not of collected
clinical data. **Treat all metrics below as a pipeline demonstration, not as
evidence of clinical validity.**


## 🎯 Target leakage: UPDRS / MoCA / Functional Assessment

UPDRS and MoCA are clinical assessment instruments *for Parkinson's* — a
clinician's score is downstream of the diagnosis, so using them as features
is close to using the label. To quantify this, two models were trained with
**identical architecture, identical hyperparameters, the same `random_state=42`
split, and the same 0.50 decision threshold**. Nothing was tuned in either arm.

### Feature association with the target (ranked, top 10 by mutual information)

| # | Feature | Pearson r | Mutual info |
|---|---|---|---|
| 1 | **UPDRS** | +0.3980 | 0.1453 |
| 2 | **FunctionalAssessment** | −0.2250 | 0.0591 |
| 3 | Rigidity | +0.1856 | 0.0437 |
| 4 | **MoCA** | −0.1731 | 0.0344 |
| 5 | Tremor | +0.2744 | 0.0295 |
| 6 | Constipation | +0.0253 | 0.0119 |
| 7 | Ethnicity | −0.0051 | 0.0114 |
| 8 | Bradykinesia | +0.1840 | 0.0097 |
| 9 | SpeechProblems | −0.0122 | 0.0096 |
| 10 | Stroke | +0.0281 | 0.0087 |

The top 5 are the three clinical scores plus two motor signs. **17 of 32
features have a mutual information of exactly 0.0000** with the target —
including Age, Gender, BMI, all four cholesterol measures, both blood
pressures, and family history. Full ranking in
[`analysis/eda_leakage.py`](analysis/eda_leakage.py) output.

### Held-out test metrics (n = 421, 62.0% positive)

Baseline to beat: **0.6200** (predict the majority class every time).

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **A** — all features | 32 | **0.8504** | 0.8694 | 0.8927 | 0.8809 | **0.9121** |
| **B** — UPDRS, MoCA, FunctionalAssessment removed | 29 | **0.6912** | 0.7148 | 0.8352 | 0.7703 | **0.7368** |
| *C* — B, also without motor signs *(supplementary)* | 24 | *0.6081* | *0.6171* | *0.9693* | *0.7541* | ***0.5302*** |

Delta, A → B: accuracy **−0.1591**, precision −0.1546, recall −0.0575,
F1 −0.1106, ROC-AUC **−0.1753**.

#### Confusion matrices (rows = actual 0/1, columns = predicted 0/1)

```text
Model A (all features)          Model B (scores removed)        Model C (supplementary)
        pred 0  pred 1                  pred 0  pred 1                  pred 0  pred 1
act 0      125      35          act 0       73      87          act 0        3     157
act 1       28     233          act 1       43     218          act 1        8     253
```

### Interpretation

**Your leakage concern is confirmed.** Three of 32 features carry most of the
model's discriminative power. Removing them costs 17.5 ROC-AUC points and
leaves accuracy at 0.6912 against a 0.6200 majority baseline — the honest,
non-clinical-assessment signal in this dataset is thin.

The supplementary Model C is the sharper result: with the motor-symptom flags
(Tremor, Rigidity, Bradykinesia, PosturalInstability, SpeechProblems) also
removed, **ROC-AUC collapses to 0.5302 — indistinguishable from random
guessing**, and the model degenerates into near-always predicting the positive
class (3 true negatives out of 160). Every bit of predictive signal in this
dataset lives in features that are *observations of Parkinson's disease
itself*, not antecedent risk factors. There is no demographic or
metabolic-risk signal to learn.

Reproduce with:

```bash
python analysis/eda_leakage.py
```

```bash
python analysis/leakage_experiment.py
```


## 🛠️ Preprocessing

- Cleaned missing and irrelevant values
- Standardized numeric features using `StandardScaler`
- Data split into training and testing sets (80/20)


## 🤖 Model

- **Algorithm**: Deep Neural Network (Keras Sequential)
- **Framework**: TensorFlow + Keras
- **Architecture**:
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Final sigmoid activation
- **Output**: Probability of Parkinson's presence
- **Artifacts**:
  - `parkinsons_model.h5`: Trained model
  - `scaler.pkl`: Fitted scaler for preprocessing


## 🧪 Prediction Script

Use the CLI script `predict_single.py` to:
- Manually input patient features
- Predict Parkinson’s risk using the saved model and scaler
- Print a clear diagnostic result


## 📎 Run Locally

```bash
git clone https://github.com/Aiza166/NeuroDetect.git
cd NeuroDetect
pip install -r requirements.txt
```


## 📂 Project Structure

```bash
NeuroDetect/
├── README.md                    # Project overview and documentation
├── requirements.txt             # Python dependencies
├── all-files.txt                # Internal file listing (optional)
│
├── data/
│   ├── raw/
│   │   └── parkinsons.csv       # Original dataset from Kaggle
│   └── processed/
│       └── parkinsons_clean.csv # Cleaned dataset used for training
│
├── disease-predictor/
│   └── Parkinson_Detection.ipynb # Notebook for training, EDA, and evaluation
│
├── models/
│   ├── parkinsons_model.h5      # Trained deep learning model
│   └── scaler.pkl               # Scaler used for feature standardization
│
├── src/
│   ├── predict_single.py        # Script for manual prediction from user input
│   │
│   ├── data_pipeline/
│   │   ├── download_data.py     # (Optional) Logic for data fetching
│   │   └── preprocess_data.py   # Data cleaning and transformation
│   │
│   └── models/
│       ├── train_model.py       # Model architecture and training pipeline
│       └── predict.py           # Model inference and evaluation script
```



## 📦 Requirements

This project uses Python 3.10 and the following libraries:

```bash
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
tensorflow==2.15.0
joblib==1.3.2
kagglehub==0.2.0
```
Install with:
```bash
pip install -r requirements.txt
```


## 🔍 Future Enhancements

- Add audio-based detection via voice samples (e.g. tremor in speech)
- Create a web frontend using Streamlit
- Dockerize the app and add CI/CD for production deployment

