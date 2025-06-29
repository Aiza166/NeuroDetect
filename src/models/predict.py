import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('models/parkinsons_model.h5')
scaler = joblib.load('models/scaler.pkl')

# Sample input (replace this with real data or user input later)
sample_data = {
    'MDVP:Fo(Hz)': 119.992,
    'MDVP:Fhi(Hz)': 157.302,
    'MDVP:Flo(Hz)': 74.997,
    'MDVP:Jitter(%)': 0.00784,
    'MDVP:Jitter(Abs)': 0.00007,
    'MDVP:RAP': 0.00370,
    'MDVP:PPQ': 0.00554,
    'Jitter:DDP': 0.01109,
    'MDVP:Shimmer': 0.04374,
    'MDVP:Shimmer(dB)': 0.426,
    'Shimmer:APQ3': 0.02182,
    'Shimmer:APQ5': 0.03130,
    'MDVP:APQ': 0.02971,
    'Shimmer:DDA': 0.06545,
    'NHR': 0.02211,
    'HNR': 21.033,
    'RPDE': 0.414783,
    'DFA': 0.815285,
    'spread1': -4.813031,
    'spread2': 0.266482,
    'D2': 2.301442,
    'PPE': 0.284654
}
# Convert sample data to DataFrame
input_df = pd.DataFrame([sample_data])

# Ensure all expected columns are present
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0][0]
result = "Parkinson's Detected" if prediction > 0.5 else "No Parkinson's"

print(f"Prediction: {result} (Confidence: {prediction:.2f})")
