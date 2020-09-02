import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
from joblib import dump
import typer
import boto3

train = typer.Typer()


@train.command()
def train(production_ready: bool = False):
    TRACKING_URI = (
        "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    s3 = boto3.client("s3")
    bucket_name = "workshop-mlflow-artifacts"
    print("Start downloading training and test data from S3...")
    s3.download_file(bucket_name, "data/train.csv", "train.csv")
    s3.download_file(bucket_name, "data/train.csv", "test.csv")
    print("Downloaded training and test data from S3!")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    X_train = train.drop(["subject", "Activity"], axis=1)
    X_test = test.drop(["subject", "Activity"], axis=1)
    y_train = train["Activity"]
    y_test = test["Activity"]
    max_depth = 9
    with mlflow.start_run(experiment_id=0):
        print(mlflow.get_artifact_uri())
        mlflow.log_param("max_depth", max_depth)
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth)
        dt_classifier.fit(X_train, y_train)
        y_pred_train = dt_classifier.predict(X_train)
        y_pred_test = dt_classifier.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        mlflow.log_metric("training accuracy", train_accuracy)
        mlflow.log_metric("test accuracy", test_accuracy)
        dump(dt_classifier, f"{os.getcwd()}/activity_classifier.joblib")
        mlflow.log_artifact(f"{os.getcwd()}/activity_classifier.joblib")
        if production_ready:
            mlflow.set_tag("production_ready", 1)
        else:
            mlflow.set_tag("production_candidate", 1)
        os.remove(f"{os.getcwd()}/activity_classifier.joblib")
        os.remove(f"{os.getcwd()}/train.csv")
        os.remove(f"{os.getcwd()}/test.csv")


if __name__ == "__main__":
    train()
