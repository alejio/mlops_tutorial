import logging
import os
from enum import Enum


class Config:
    # Your config! Go ahead and edit with your own values.

    USER = "alejio"  # UPDATE THIS WITH YOUR GITHUB HANDLE IN THE AlmostOps STAGE OF THE WORKSHOP
    EXPERIMENT_ID = "2"  # update with what mlflow_setup.py returns

    # !!Leave the below variables as they are!!
    EXPERIMENT_NAME = f"{USER}_predict_sentiment"
    BUCKET_NAME = "workshop-mlflow-artifacts"  # update with your own bucket
    TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    LIVE_TAG = "live"
    CANDIDATE_TAG = "production_candidate"
    LOGGING = logging.INFO
    S3_DATA_DIR = "data"
    S3_ARTIFACTS_DIR = f"almostops_artifacts/{USER}"
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    FEATURE_ENGINEERING_ARTIFACT = "feature_engineering.joblib"
    CLASSIFIER_ARTIFACT = "classifier.joblib"
    LOCAL_ARTIFACTS_PATH = "artifacts"


class ArtifactLocation(Enum):
    LOCAL = "local"
    S3 = "s3"
    S3_MLFLOW = "s3_mlflow"
