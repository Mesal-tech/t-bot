
import joblib
import os
import sys

print("Testing model loading...")
model_path = 'models/saved/EURUSD_model.pkl'

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

try:
    print(f"Loading {model_path}...")
    data = joblib.load(model_path)
    print("Keys:", data.keys())
    if 'model' in data:
        print("Model object loaded.")
    if 'scaler' in data:
        print("Scaler loaded.")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED to load model: {e}")
    sys.exit(1)
