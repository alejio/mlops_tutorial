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
from config import Config
from typing import Dict


logging.basicConfig(level=Config.LOGGING)


def download_and_load_data(
    bucket_name: str, directory: str, train_csv: str, test_csv: str
) -> Dict:

    s3 = boto3.client("s3")
    bucket_name = bucket_name
    # TODO: pandas read directly from s3
    logging.info("Start downloading training and test data from S3...")
    s3.download_file(bucket_name, f"{directory}/{train_csv}", train_csv)
    s3.download_file(bucket_name, f"{directory}/{test_csv}", test_csv)
    logging.info("Downloaded training and test data from S3!")

    train = pd.read_csv(train_csv, names=["review", "sentiment"])
    test = pd.read_csv(test_csv, names=["review", "sentiment"])

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


def train(production_ready: bool = False) -> None:

    mlflow.set_tracking_uri(Config.TRACKING_URI)
    data_dict = download_and_load_data(
        Config.BUCKET_NAME, Config.S3_DATA_DIR, Config.TRAIN_CSV, Config.TEST_CSV
    )

    with mlflow.start_run(experiment_id=Config.EXPERIMENT_ID):
        logging.info(mlflow.get_artifact_uri())

        feature_engineering_params = {"binary": True}
        for k, v in feature_engineering_params.items():
            mlflow.log_param(str(k), str(v))
        feature_engineering = CountVectorizer(**feature_engineering_params)

        classifier_params = {"alpha": 0.75, "binarize": 0.0}
        for k, v in classifier_params.items():
            mlflow.log_param(str(k), str(v))
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

        mlflow.log_metric("training accuracy", train_accuracy)
        mlflow.log_metric("test accuracy", test_accuracy)

        logging.info("Persisting models..")
        dump(feature_engineering, f"{os.getcwd()}/feature_engineering.joblib")
        mlflow.log_artifact(f"{os.getcwd()}/feature_engineering.joblib")

        dump(classifier, f"{os.getcwd()}/classifier.joblib")
        mlflow.log_artifact(f"{os.getcwd()}/classifier.joblib")
        logging.info("Done persisting models!")

        if production_ready:
            mlflow.set_tag(Config.LIVE_TAG, 1)
        else:
            mlflow.set_tag(Config.CANDIDATE_TAG, 1)

        # Cleanup
        os.remove(f"{os.getcwd()}/feature_engineering.joblib")
        os.remove(f"{os.getcwd()}/classifier.joblib")
        os.remove(f"{os.getcwd()}/train.csv")
        os.remove(f"{os.getcwd()}/test.csv")


if __name__ == "__main__":
    typer.run(train)
