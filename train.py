import logging
import os
from typing import Dict

import boto3
import mlflow
import typer
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

from config import Config, ArtifactLocation
from utils import load_and_preprocess_data

logging.basicConfig(level=Config.LOGGING)


def train_and_persist(data_dict: Dict) -> Dict:
    # Model config
    feature_engineering_params = {"binary": True}
    feature_engineering = CountVectorizer(**feature_engineering_params)
    classifier_params = {"alpha": 0.79, "binarize": 0.0}
    classifier = BernoulliNB(**classifier_params)

    # Model training
    logging.info("Begin training..")
    X_train = feature_engineering.fit_transform(data_dict["X_raw_train"])
    classifier.fit(X_train, data_dict["y_train"])
    logging.info("Done training!")

    # Performance metrics
    y_pred_train = classifier.predict(X_train)
    train_accuracy = accuracy_score(data_dict["y_train"], y_pred_train)
    X_test = feature_engineering.transform(data_dict["X_raw_test"])
    y_pred_test = classifier.predict(X_test)
    test_accuracy = accuracy_score(data_dict["y_test"], y_pred_test)
    logging.info(
        f"Training - Test accuracy: {round(100 * train_accuracy, 2)}% - {round(100 * test_accuracy, 2)}%"
    )

    # Persist
    logging.info("Persisting models..")
    dump(
        feature_engineering,
        f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.FEATURE_ENGINEERING_ARTIFACT}",
    )
    dump(
        classifier,
        f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.CLASSIFIER_ARTIFACT}",
    )
    logging.info("Done persisting models!")

    return {
        "params": {
            "feature_engineering": feature_engineering_params,
            "classifier": classifier_params,
        },
        "accuracy": {"train": train_accuracy, "test": test_accuracy},
    }


def main(artifact_location: str, production_ready: bool = False) -> None:
    art_loc = ArtifactLocation(artifact_location)
    data_dict = load_and_preprocess_data(art_loc)

    if art_loc == ArtifactLocation.LOCAL:
        _ = train_and_persist(data_dict)

    elif art_loc == ArtifactLocation.S3:
        _ = train_and_persist(data_dict)
        s3 = boto3.client("s3")
        s3.upload_file(
            f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.FEATURE_ENGINEERING_ARTIFACT}",
            Bucket=Config.BUCKET_NAME,
            Key=f"{Config.S3_ARTIFACTS_DIR}/{Config.FEATURE_ENGINEERING_ARTIFACT}",
        )
        s3.upload_file(
            f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.CLASSIFIER_ARTIFACT}",
            Bucket=Config.BUCKET_NAME,
            Key=f"{Config.S3_ARTIFACTS_DIR}/{Config.CLASSIFIER_ARTIFACT}",
        )

    elif ArtifactLocation.S3_MLFLOW:
        mlflow.set_tracking_uri(Config.TRACKING_URI)

        # MLflow experiment tracking
        with mlflow.start_run(experiment_id=Config.EXPERIMENT_ID):
            training_metadata = train_and_persist(data_dict)
            logging.info(mlflow.get_artifact_uri())
            for k, v in training_metadata["params"]["feature_engineering"].items():
                mlflow.log_param(str(k), str(v))
            for k, v in training_metadata["params"]["classifier"].items():
                mlflow.log_param(str(k), str(v))

            mlflow.log_metric(
                "training accuracy", training_metadata["accuracy"]["train"]
            )
            mlflow.log_metric("test accuracy", training_metadata["accuracy"]["test"])
            mlflow.log_artifact(
                f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.FEATURE_ENGINEERING_ARTIFACT}"
            )
            mlflow.log_artifact(
                f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/{Config.CLASSIFIER_ARTIFACT}"
            )

            if production_ready:
                mlflow.set_tag(Config.LIVE_TAG, 1)
            else:
                mlflow.set_tag(Config.LIVE_TAG, 0)
                mlflow.set_tag(Config.CANDIDATE_TAG, 1)

            # When running in Github actions set EXPERIMENT_ID as env
            # for consumption by the subsequent step
            print(f"::set-output name=EXPERIMENT_ID::{Config.EXPERIMENT_ID}")


if __name__ == "__main__":
    typer.run(main)
