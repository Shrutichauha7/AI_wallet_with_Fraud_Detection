import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
import yaml
from dvclive import Live
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('fraud_model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'fraud_evaluation.log'))

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)

    if params is None:
        raise ValueError("params.yaml is empty")

    return params


def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)

    logger.debug(f"Model loaded from {file_path}")
    return model


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.debug(f"Data loaded from {file_path}, shape: {df.shape}")
    return df


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)

    logger.debug(f"Metrics: {metrics}")
    logger.debug(f"Confusion Matrix:\n{cm}")

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:\n", cm)

    return metrics


def save_metrics(metrics: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)

    logger.debug(f"Metrics saved at {file_path}")


def main():
    try:
        # Load params
        params = load_params('params.yaml')

        # Load model
        model = load_model('./models/xgb_model.pkl')

        # Load test data
        test_data = load_data('./data/processed/test_processed.csv')

        # Split features & target
        X_test = test_data.drop(columns=['Class']).values
        y_test = test_data['Class'].values

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log with DVC Live
        with Live(save_dvc_exp=True) as live:
            for key, value in metrics.items():
                live.log_metric(key, value)

            live.log_params(params)

        # Save metrics
        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()