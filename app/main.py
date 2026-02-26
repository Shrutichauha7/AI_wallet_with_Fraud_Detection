from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from app.model_loader import predict_fraud

app = FastAPI(title="AI Wallet Fraud Detection API")


class Transaction(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "AI Wallet Fraud Detection Running"}


@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_fraud(transaction.features)
    return result


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)