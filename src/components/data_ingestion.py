import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj2_data import Proj2Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            my_data = Proj2Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise MyException(e,sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method")

        try:
            # ✅ Load schema
            from src.constants import SCHEMA_FILE_PATH
            from src.utils.main_utils import read_yaml_file

            schema = read_yaml_file(SCHEMA_FILE_PATH)
            target_column = schema["target_column"]

            # 🔴 STEP 1: Check NaN in target
            nan_count = dataframe[target_column].isna().sum()
            logging.info(f"NaN count in target before cleaning: {nan_count}")

            # 🔴 STEP 2: Drop NaN rows
            dataframe = dataframe.dropna(subset=[target_column])

            # 🔴 STEP 3: Convert target to int
            dataframe[target_column] = dataframe[target_column].astype(int)

            # 🔍 STEP 4: Check class distribution
            class_counts = dataframe[target_column].value_counts()
            logging.info("Class distribution after cleaning:")
            logging.info(f"\n{class_counts}")

            # 🚨 SAFETY CHECKS
            if len(class_counts) < 2:
                raise Exception("Dataset has only one class after cleaning")

            if class_counts.min() < 2:
                raise Exception("Not enough minority samples for stratified split")

            # 🔴 STEP 5: Stratified split (CRITICAL)
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=dataframe[target_column],
                random_state=42
            )

            logging.info("Performed stratified train-test split")

            # 🔍 Verify split
            logging.info("Train class distribution:")
            logging.info(f"\n{train_set[target_column].value_counts()}")

            logging.info("Test class distribution:")
            logging.info(f"\n{test_set[target_column].value_counts()}")

            # ✅ Save files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test file path")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e