import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded from {data_url}, shape: {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess fraud detection data."""
    try:
        
        drop_cols = ['Unnamed: 0', 'TransactionID']  
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        
        if 'Class' not in df.columns:
            raise KeyError("Target column 'Class' not found in dataset")

        df.fillna(0, inplace=True)

        logger.debug("Data preprocessing completed, shape: {}".format(df.shape))
        return df
    except KeyError as e:
        logger.error(f"Missing column: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Train and test data saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main():
    try:
        
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        data_url = 'C:\practice\Fraud_Detection_Analysis\experiments\creditdata.csv' 

        df = load_data(data_url)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42, stratify=final_df['Class'])
        save_data(train_data, test_data, data_path='./data')
        logger.info("Data ingestion process completed successfully")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()