import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_df(df, target_column='Class'):
    """
    Preprocess fraud dataset:
    - Handle missing values
    - Encode categorical columns
    - Scale numerical columns
    - Remove duplicates
    """
    try:
        logger.debug("Starting preprocessing")

        # Drop duplicates
        df = df.drop_duplicates()
        logger.debug("Duplicates removed")

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        logger.debug("Missing values handled")

        # Separate target
        X = df.drop("Class", axis=1)
        y = df["Class"]

        

        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        logger.debug("Numerical features scaled")

        # Combine back
        df_processed = pd.concat([X, y], axis=1)

        return df_processed

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def main(target_column='Class'):
    try:
        # Load data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        logger.debug("Data loaded successfully")

        # Preprocess
        train_processed = preprocess_df(train_data, target_column)
        test_processed = preprocess_df(test_data, target_column)

        # Save processed data
        output_path = './data/processed'
        os.makedirs(output_path, exist_ok=True)

        train_processed.to_csv(os.path.join(output_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(output_path, 'test_processed.csv'), index=False)

        logger.debug("Processed data saved successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()