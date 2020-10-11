import logging
from enum import Enum
import os


class Config:
    EXPERIMENT_ID = "2"
    EXPERIMENT_NAME = "predict_sentiment"
    TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]  # Set as environment variable
    LIVE_TAG = "live"
    CANDIDATE_TAG = "production_candidate"
    BUCKET_NAME = "workshop-mlflow-artifacts"  # Update with your own!
    LOGGING = logging.INFO
    S3_DATA_DIR = "data"
    S3_ARTIFACTS_DIR = "artifacts"
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    FEATURE_ENGINEERING_ARTIFACT = "feature_engineering.joblib"
    CLASSIFIER_ARTIFACT = "classifier.joblib"
    LOCAL_ARTIFACTS_PATH = "artifacts"
    LOCAL_ARTIFACTS_PATH_FROM_S3 = "artifacts_s3"


class ArtifactLocation(Enum):
    LOCAL = "local"
    S3 = "s3"
    S3_MLFLOW = "s3_mlflow"
