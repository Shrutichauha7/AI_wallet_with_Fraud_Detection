from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import model_loader

app = FastAPI(title="AI Wallet Fraud Detection API")



class Transaction(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: int
    risk_level: str


@app.get("/")
def home():
    return {"message": "AI Wallet Fraud Detection Running 🚀"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        return model_loader.predict_fraud(transaction.features)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)