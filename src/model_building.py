import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file safely."""
    try:
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"{params_path} does not exist")

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        # ✅ Handle empty YAML
        if params is None:
            raise ValueError(f"{params_path} is empty or invalid")

        # ✅ Validate required section
        if 'model_building' not in params:
            raise KeyError("'model_building' key not found in params.yaml")

        logger.debug(f"Parameters successfully loaded from {params_path}")
        return params

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise

    except KeyError as e:
        logger.error(f"Missing key in YAML: {e}")
        raise

    except ValueError as e:
        logger.error(f"Invalid YAML content: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_model(X_train, y_train, params):
    try:
        # Handle class imbalance using scale_pos_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))

        scale_pos_weight = weight_dict[1] / weight_dict[0]
        logger.debug(f"scale_pos_weight: {scale_pos_weight}")

        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=params['random_state'],
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )

        logger.debug("Training XGBoost model...")
        model.fit(X_train, y_train)
        logger.debug("Model training completed")

        return model

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved at {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    try:
        
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/interim/train_processed.csv')
        
        X_train = train_data.drop(columns=['Class']).values
        y_train = train_data['Class'].values

        
        model = train_model(X_train, y_train, params)
        model_save_path = 'models/model.pkl'
        save_model(model, 'models/xgb_model.pkl')

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()