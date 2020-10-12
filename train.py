import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import mlflow
from joblib import dump
import typer
import boto3
import logging
from config import Config, ArtifactLocation
from typing import Dict, Optional
from utils import get_full_s3_path


logging.basicConfig(level=Config.LOGGING)


def load_csv_to_pandas(
    artifact_location: ArtifactLocation,
    bucket_name: Optional[str],
    directory: Optional[str],
    train_csv: str,
    test_csv: str,
):
    if artifact_location == ArtifactLocation.LOCAL:
        path_train = f"data/{train_csv}"
        path_test = f"data/{test_csv}"
    else:
        path_train = get_full_s3_path(bucket_name, Config.S3_DATA_DIR, train_csv)
        path_test = get_full_s3_path(bucket_name, Config.S3_DATA_DIR, test_csv)
    train = pd.read_csv(path_train, names=["review", "sentiment"])
    test = pd.read_csv(path_test, names=["review", "sentiment"])
    return train, test


def load_and_preprocess_data(artifact_location: ArtifactLocation) -> Dict:
    train, test = load_csv_to_pandas(
        artifact_location,
        Config.BUCKET_NAME,
        Config.S3_DATA_DIR,
        Config.TRAIN_CSV,
        Config.TEST_CSV,
    )

    X_raw_train = train["review"]
    X_raw_test = test["review"]
    y_train = train["sentiment"]
    y_test = test["sentiment"]
    return {
        "X_raw_train": X_raw_train,
        "X_raw_test": X_raw_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train(artifact_location: str, production_ready: bool = False) -> None:

    art_loc = ArtifactLocation(artifact_location)
    data_dict = load_and_preprocess_data(art_loc)

    if art_loc == ArtifactLocation.LOCAL:
        feature_engineering_params = {"binary": True}
        feature_engineering = CountVectorizer(**feature_engineering_params)
        classifier_params = {"alpha": 0.75, "binarize": 0.0}
        classifier = BernoulliNB(**classifier_params)

        logging.info("Begin training..")
        X_train = feature_engineering.fit_transform(data_dict["X_raw_train"])
        classifier.fit(X_train, data_dict["y_train"])
        y_pred_train = classifier.predict(X_train)
        train_accuracy = accuracy_score(data_dict["y_train"], y_pred_train)
        logging.info("Done training!")

        X_test = feature_engineering.transform(data_dict["X_raw_test"])
        y_pred_test = classifier.predict(X_test)
        test_accuracy = accuracy_score(data_dict["y_test"], y_pred_test)
        logging.info(
            f"Training - Test accuracy: {round(100*train_accuracy, 2)}% - {round(100*test_accuracy, 2)}%"
        )
        logging.info("Persisting models..")
        dump(
            feature_engineering,
            f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/feature_engineering.joblib",
        )
        dump(
            classifier, f"{os.getcwd()}/{Config.LOCAL_ARTIFACTS_PATH}/classifier.joblib"
        )
        logging.info("Done persisting models!")

    elif art_loc == ArtifactLocation.S3:
        feature_engineering_params = {"binary": True}
        feature_engineering = CountVectorizer(**feature_engineering_params)
        classifier_params = {"alpha": 0.75, "binarize": 0.0}
        classifier = BernoulliNB(**classifier_params)

        logging.info("Begin training..")
        X_train = feature_engineering.fit_transform(data_dict["X_raw_train"])
        classifier.fit(X_train, data_dict["y_train"])
        y_pred_train = classifier.predict(X_train)
        train_accuracy = accuracy_score(data_dict["y_train"], y_pred_train)
        logging.info("Done training!")

        X_test = feature_engineering.transform(data_dict["X_raw_test"])
        y_pred_test = classifier.predict(X_test)
        test_accuracy = accuracy_score(data_dict["y_test"], y_pred_test)
        logging.info(
            f"Training - Test accuracy: {round(100*train_accuracy, 2)}% - {round(100*test_accuracy, 2)}%"
        )

        logging.info("Persisting models..")
        dump(feature_engineering, f"{os.getcwd()}/feature_engineering.joblib")
        dump(classifier, f"{os.getcwd()}/classifier.joblib")
        logging.info("Done persisting models!")
        s3 = boto3.client("s3")
        s3.upload_file(
            f"{os.getcwd()}/feature_engineering.joblib",
            Bucket=Config.BUCKET_NAME,
            Key=f"{Config.S3_ARTIFACTS_DIR}/{Config.FEATURE_ENGINEERING_ARTIFACT}",
        )
        s3.upload_file(
            f"{os.getcwd()}/classifier.joblib",
            Bucket=Config.BUCKET_NAME,
            Key=f"{Config.S3_ARTIFACTS_DIR}/{Config.CLASSIFIER_ARTIFACT}",
        )

    elif ArtifactLocation.S3_MLFLOW:
        mlflow.set_tracking_uri(Config.TRACKING_URI)
        with mlflow.start_run(experiment_id=Config.EXPERIMENT_ID):
            feature_engineering_params = {"binary": True}
            feature_engineering = CountVectorizer(**feature_engineering_params)
            classifier_params = {"alpha": 0.75, "binarize": 0.0}
            classifier = BernoulliNB(**classifier_params)

            logging.info("Begin training..")
            X_train = feature_engineering.fit_transform(data_dict["X_raw_train"])
            classifier.fit(X_train, data_dict["y_train"])
            y_pred_train = classifier.predict(X_train)
            train_accuracy = accuracy_score(data_dict["y_train"], y_pred_train)
            logging.info("Done training!")

            X_test = feature_engineering.transform(data_dict["X_raw_test"])
            y_pred_test = classifier.predict(X_test)
            test_accuracy = accuracy_score(data_dict["y_test"], y_pred_test)
            logging.info(
                f"Training - Test accuracy: {round(100*train_accuracy, 2)}% - {round(100*test_accuracy, 2)}%"
            )
            logging.info(mlflow.get_artifact_uri())
            for k, v in feature_engineering_params.items():
                mlflow.log_param(str(k), str(v))
            for k, v in classifier_params.items():
                mlflow.log_param(str(k), str(v))
            mlflow.log_metric("training accuracy", train_accuracy)
            mlflow.log_metric("test accuracy", test_accuracy)
            mlflow.log_artifact(f"{os.getcwd()}/feature_engineering.joblib")
            mlflow.log_artifact(f"{os.getcwd()}/classifier.joblib")

            if production_ready:
                mlflow.set_tag(Config.LIVE_TAG, 1)
            else:
                mlflow.set_tag(Config.CANDIDATE_TAG, 1)


if __name__ == "__main__":
    typer.run(train)
