import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
print("Loading model and scaler...")
model = load_model('models/parkinsons_model.h5')
scaler = joblib.load('models/scaler.pkl')

# Load dataset to get correct feature names
df = pd.read_csv('data/processed/parkinsons_clean.csv')

# Drop only the columns that exist
columns_to_drop = [col for col in ['Diagnosis', 'PatientID', 'DoctorInCharge'] if col in df.columns]
feature_columns = df.drop(columns=columns_to_drop).columns.tolist()

# Dummy values for prediction input
# Replace these with real values (e.g. extracted from an audio file later)
sample_input = [0.5] * len(feature_columns)  # 32 features typically

# Create DataFrame with correct shape and columns
input_df = pd.DataFrame([sample_input], columns=feature_columns)

input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0][0]
label = "Parkinson's Detected" if prediction > 0.5 else "No Parkinson's"

# Output
print(f"Prediction Probability: {prediction:.4f}")
print(f"Result: {label}")

# This code demonstrates how to load a pre-trained model and scaler,
# prepare a single input sample, and make a prediction.
# Note: The input features should be replaced with actual values extracted from audio data.
# This is a placeholder to demonstrate the prediction process.
