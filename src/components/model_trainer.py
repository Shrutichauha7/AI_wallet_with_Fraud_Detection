import os
import sys
import pickle
import numpy as np
from xgboost import XGBClassifier

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score
from src.entity.artifact_entity import ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            # 🔍 Debug
            unique_classes = np.unique(y_train)
            logging.info(f"Unique classes in y_train: {unique_classes}")

            # 🚨 Ensure binary classification
            if not set(unique_classes).issubset({0, 1}):
                raise Exception(f"Expected binary classes [0,1], got {unique_classes}")

            # ✅ Compute scale_pos_weight
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)

            if pos_count == 0:
                raise Exception("No positive class (1) found in training data")

            scale_pos_weight = neg_count / pos_count
            logging.info(f"scale_pos_weight: {scale_pos_weight}")

            # ✅ Model
            model = XGBClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                learning_rate=self.model_trainer_config.learning_rate,
                max_depth=self.model_trainer_config.max_depth,
                random_state=self.model_trainer_config.random_state,
                eval_metric=self.model_trainer_config.eval_metric,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False
            )

            model.fit(X_train, y_train)
            logging.info("Model training completed")

            return model

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        
        try:
            logging.info("Starting model training")

            # 🔴 Load data
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            # 🔴 Step 1: Remove NaN
            nan_count = np.isnan(y_train).sum()
            logging.info(f"NaN count in y_train: {nan_count}")

            if nan_count > 0:
                mask = ~np.isnan(y_train)
                X_train = X_train[mask]
                y_train = y_train[mask]

            # 🔴 Step 2: Convert to int
            y_train = y_train.astype(int)

            # 🔍 Step 3: Check class distribution BEFORE SMOTE
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logging.info(f"Before SMOTE: {class_dist}")

            # 🚨 Safety check
            if len(unique) < 2:
                raise Exception("Only one class present. Cannot train model.")

            # 🔴 Step 4: Apply SMOTE (SAFE VERSION)
            minority_count = min(counts)

            if minority_count < 2:
                raise Exception("Not enough minority samples for SMOTE")

            # ⚠️ Adjust k_neighbors dynamically
            k_neighbors = min(5, minority_count - 1)

            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # 🔍 After SMOTE
            unique, counts = np.unique(y_train, return_counts=True)
            logging.info(f"After SMOTE: {dict(zip(unique, counts))}")

            # 🔴 Step 5: Train model
            model = self.train_model(X_train, y_train)

            # 🔴 Step 6: Load test data
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # 🔴 Clean test target
            mask = ~np.isnan(y_test)
            X_test = X_test[mask]
            y_test = y_test[mask].astype(int)

            # 🔴 Step 7: Predictions
            y_pred = model.predict(X_test)

            # 🔴 Step 8: Metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            logging.info(f"F1 Score: {f1}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")

            # 🔴 Step 9: Create metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            # 🔴 Step 10: Save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            logging.info(
                f"Model saved at {self.model_trainer_config.trained_model_file_path}"
            )

            # 🔴 Step 11: Return artifact (FIXED)
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
        except Exception as e: 
            raise MyException(e, sys)