import logging


class Config:
    EXPERIMENT_ID = "2"
    TRACKING_URI = (
        "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
    )
    LIVE_TAG = "live"
    CANDIDATE_TAG = "production_candidate"
    BUCKET_NAME = "workshop-mlflow-artifacts"
    LOGGING = logging.INFO
    S3_DATA_DIR = "data"
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
