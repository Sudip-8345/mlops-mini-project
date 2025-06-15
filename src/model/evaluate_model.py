import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import mlflow.sklearn
import logging
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# MLflow and DagsHub tracking
import dagshub
dagshub.init(repo_owner='Sudip-8345', repo_name='DVC-git-mini-Project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/DVC-git-mini-Project.mlflow")
mlflow.set_experiment("Model Evaluation")

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Evaluation error: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Saving metrics failed: %s', e)
        raise

def main():
    with mlflow.start_run(run_name="Model Evaluation with SVC"):
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_tfidf.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Save metrics for DVC tracking
            save_metrics(metrics, 'reports/metrics.json')

            logger.info('Evaluation completed successfully.')

        except Exception as e:
            logger.error('Pipeline failed: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
