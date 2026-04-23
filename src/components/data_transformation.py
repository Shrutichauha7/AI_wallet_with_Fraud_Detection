import sys
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_object
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.constants import SCHEMA_FILE_PATH


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact,
                 data_validation_artifact,
                 data_transformation_config: DataTransformationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)

    def get_preprocessor_object(self):
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            num_features = self._schema_config["num_features"]   # Time
            mm_columns = self._schema_config["mm_columns"]       # Amount

            remaining_cols = [col for col in numerical_columns if col not in num_features]

            # Pipelines
            standard_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            minmax_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler())
            ])

            passthrough_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ])

            preprocessor = ColumnTransformer([
                ("standard_pipeline", standard_pipeline, num_features),
                ("minmax_pipeline", minmax_pipeline, mm_columns),
                ("passthrough_pipeline", passthrough_pipeline, remaining_cols)
            ])

            return preprocessor

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Cannot proceed.")

            # Load data
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            target_column = self._schema_config["target_column"]

            # 🔴 STEP 1: Remove NaN targets
            train_df = train_df.dropna(subset=[target_column])
            test_df = test_df.dropna(subset=[target_column])

            logging.info(f"Train shape after cleaning: {train_df.shape}")
            logging.info(f"Test shape after cleaning: {test_df.shape}")

            # 🔴 STEP 2: Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # 🔴 STEP 3: Enforce binary target
            target_feature_train_df = target_feature_train_df.astype(int)
            target_feature_test_df = target_feature_test_df.astype(int)

            # 🔍 Validate target values
            train_unique = target_feature_train_df.unique()
            test_unique = target_feature_test_df.unique()

            logging.info(f"Train target unique: {train_unique}")
            logging.info(f"Test target unique: {test_unique}")

            if not set(train_unique).issubset({0, 1}):
                raise Exception(f"Invalid train target values: {train_unique}")

            if not set(test_unique).issubset({0, 1}):
                raise Exception(f"Invalid test target values: {test_unique}")

            # 🔴 STEP 4: Preprocessing
            preprocessor = self.get_preprocessor_object()

            logging.info("Fitting preprocessor")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            logging.info("Transforming test data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # 🔴 STEP 5: Combine features + target safely
            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_df.values.reshape(-1, 1)]

            test_arr = np.c_[input_feature_test_arr,
                             target_feature_test_df.values.reshape(-1, 1)]

            # Save paths
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            # Save arrays
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )

            logging.info("Data Transformation Completed Successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            raise MyException(e, sys)