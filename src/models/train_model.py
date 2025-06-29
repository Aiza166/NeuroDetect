import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

print("Loading processed dataset...")
data_path = 'data/processed/parkinsons_clean.csv'
df = pd.read_csv(data_path)

# Drop only if columns exist
drop_cols = ['Diagnosis', 'PatientID', 'DoctorInCharge']
existing_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(existing_cols, axis=1)
y = df['Diagnosis']

print("Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# Build neural network model
print("Building the model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("Evaluating on test set...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

model.save('models/parkinsons_model.h5')
print("Model saved to models/parkinsons_model.h5")
