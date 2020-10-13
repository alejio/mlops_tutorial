import mlflow
import logging
from config import Config, ArtifactLocation
import boto3
from typing import Tuple, Optional, Dict
import os.path
from joblib import load
import pandas as pd


def get_mlflow_run(tracking_uri: str, experiment_id: str, live_tag: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    query = f"tags.{live_tag}='1'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["attributes.end_time DESC"],
    )
    assert len(runs) > 0, "MLflow query failed to find a run!"
    return runs["run_id"].values[0]


def get_s3_path(experiment_name: str, run_id: str) -> str:
    return f"{experiment_name}/{run_id}/artifacts"


def get_full_s3_path(bucket_name: str, directory: str, fname: str) -> str:
    return f"s3://{bucket_name}/{directory}/{fname}"


def download_artifacts_from_s3(bucket_name: str, s3_path: str, local_path: str) -> None:
    logging.info(f"Bucket Name: {Config.BUCKET_NAME}")
    logging.info(f"Artifact path: {s3_path}")
    feature_engineering_path = f"{s3_path}/{Config.FEATURE_ENGINEERING_ARTIFACT}"
    logging.info(f"Feature engineering path: {feature_engineering_path}")
    classifier_path = f"{s3_path}/{Config.CLASSIFIER_ARTIFACT}"
    logging.info(f"Classifier path: {classifier_path}")
    s3 = boto3.client("s3")
    logging.info("Download start")
    feature_engineering_local_path = (
        f"{local_path}/{Config.FEATURE_ENGINEERING_ARTIFACT}"
    )
    logging.info(f"Feature engineering local path: {feature_engineering_local_path}")
    classifier_local_path = f"{local_path}/{Config.CLASSIFIER_ARTIFACT}"
    logging.info(f"Classifier local path: {classifier_local_path}")
    s3.download_file(
        Config.BUCKET_NAME,
        feature_engineering_path,
        feature_engineering_local_path,
    )
    s3.download_file(bucket_name, classifier_path, classifier_local_path)
    logging.info("Downloaded!")


def load_artifacts(artifact_location: ArtifactLocation) -> Tuple:

    if artifact_location != ArtifactLocation.LOCAL:
        if artifact_location == ArtifactLocation.S3:
            s3_path = Config.S3_ARTIFACTS_DIR
        else:
            run_id = get_mlflow_run(
                Config.TRACKING_URI, Config.EXPERIMENT_ID, Config.LIVE_TAG
            )
            s3_path = get_s3_path(Config.EXPERIMENT_NAME, run_id)
        download_artifacts_from_s3(
            Config.BUCKET_NAME, s3_path, Config.LOCAL_ARTIFACTS_PATH
        )

    feature_engineering_path = (
        f"{Config.LOCAL_ARTIFACTS_PATH}/{Config.FEATURE_ENGINEERING_ARTIFACT}"
    )
    classifier_path = f"{Config.LOCAL_ARTIFACTS_PATH}/{Config.CLASSIFIER_ARTIFACT}"
    assert os.path.isfile(
        feature_engineering_path
    ), "Feature engineering artifact not available!"
    assert os.path.isfile(classifier_path), "Classification artifact not available!"

    feature_engineering = load(feature_engineering_path)
    classifier = load(classifier_path)

    return feature_engineering, classifier


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
