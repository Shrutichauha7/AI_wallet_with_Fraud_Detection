from xgboost import XGBClassifier
import numpy as np

# Load model
model = XGBClassifier()
model.load_model(r"C:\Users\Lenovo\PycharmProjects\AI_digital_wallet\app\xgb_fraud_model.json")

def predict_fraud(features: list):
    if len(features) != 31:
        raise ValueError("Expected 31 features")

    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)  # if scaling used

    probability = model.predict_proba(data)[0][1]
    prediction = int(probability > 0.2)  # better threshold for fraud

    return {
        "fraud_probability": float(probability),
        "is_fraud": prediction
    }