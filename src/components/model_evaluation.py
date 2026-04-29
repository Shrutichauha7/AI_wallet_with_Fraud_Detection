import sys
import os
import pickle
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact
)
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            # 🔥 ROC-AUC (important metric)
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            except:
                roc_auc = 0.0

            logging.info(
                f"Metrics → F1: {f1}, Precision: {precision}, Recall: {recall}, ROC-AUC: {roc_auc}"
            )

            return f1, precision, recall, roc_auc

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")

            # 🔴 Load newly trained model
            with open(self.model_trainer_artifact.trained_model_file_path, "rb") as f:
                new_model = pickle.load(f)

            # 🔴 Load test data
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # 🔴 Clean NaN
            mask = ~np.isnan(y_test)
            X_test = X_test[mask]
            y_test = y_test[mask].astype(int)

            # 🔴 Evaluate new model
            new_f1, precision, recall, roc_auc = self.evaluate_model(new_model, X_test, y_test)

            best_model_path = self.model_eval_config.best_model_file_path

            improvement_score = 0.0
            is_model_accepted = False
            old_f1 = None

            # 🔥 Case 1: No existing model
            if not os.path.exists(best_model_path):
                logging.info("No existing model found. Accepting new model.")
                is_model_accepted = True
                improvement_score = new_f1

            else:
                # 🔴 Load old model
                with open(best_model_path, "rb") as f:
                    old_model = pickle.load(f)

                old_pred = old_model.predict(X_test)
                old_f1 = f1_score(y_test, old_pred)

                logging.info(f"Old F1: {old_f1}, New F1: {new_f1}")

                # 🔴 Calculate improvement
                improvement_score = new_f1 - old_f1

                # 🔥 Decision
                if improvement_score > self.model_eval_config.improvement_threshold:
                    is_model_accepted = True
                    logging.info("New model is better → Accepted ✅")
                else:
                    logging.info("New model is NOT better → Rejected ❌")

            # 🔴 Save evaluation report
            os.makedirs(
                os.path.dirname(self.model_eval_config.report_file_path),
                exist_ok=True
            )

            with open(self.model_eval_config.report_file_path, "w") as f:
                f.write(f"New F1 Score: {new_f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"ROC-AUC: {roc_auc}\n")
                f.write(f"Old F1 Score: {old_f1}\n")
                f.write(f"Improvement Score: {improvement_score}\n")
                f.write(f"Accepted: {is_model_accepted}\n")

            logging.info(
                f"Evaluation report saved at {self.model_eval_config.report_file_path}"
            )

            # 🔴 Return artifact
            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improvement_score=improvement_score,
                best_model_path=best_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                report_file_path=self.model_eval_config.report_file_path
            )

        except Exception as e:
            raise MyException(e, sys)