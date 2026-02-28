from xgboost import XGBClassifier
import numpy as np

# Load model
model = XGBClassifier()
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, r"C:\Users\Lenovo\PycharmProjects\AI_wallet_with_Fraud_Detection\app\xgb_fraud_model.json")

model.load_model(model_path)

EXPECTED_FEATURES = 30

def predict_fraud(features: list):

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features (V1–V28)")

    data = np.array(features, dtype=float).reshape(1, -1)

    probability = model.predict_proba(data)[0][1]

    # Lower threshold for fraud detection (better recall)
    prediction = int(probability > 0.2)

    # Risk categorization
    if probability > 0.8:
        risk_level = "HIGH RISK"
    elif probability > 0.4:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "fraud_probability": round(float(probability), 6),
        "is_fraud": prediction,
        "risk_level": risk_level
    }