# 🧠 NeuroDetect – Parkinson’s Disease Detection

Predicting the likelihood of Parkinson’s Disease using clinical and demographic data with deep learning.


## 📁 Dataset

- **Source**: [Kaggle Parkinson’s Dataset](https://www.kaggle.com/datasets/nidaguler/parkinsons-dataset)
- **Target**: `Parkinsons` (Binary classification: 0 or 1)
- **Features**: Age, Gender, Sleep Quality, BMI, Family History, UPDRS, MoCA, Functional Assessment, and more.


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

