import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
from joblib import dump
import typer

train = typer.Typer()


@train.command()
def train(production_ready: bool = False):
    TRACKING_URI = (
        "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    # client = mlflow.tracking.MlflowClient(TRACKING_URI)
    train = pd.read_csv("data/train.csv")
    X_train = train.drop(["subject", "Activity"], axis=1)
    y_train = train.Activity
    max_depth = 8
    with mlflow.start_run(experiment_id=0):
        print(mlflow.get_artifact_uri())
        mlflow.log_param("max_depth", max_depth)
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth)
        dt_classifier.fit(X_train, y_train)
        y_pred_train = dt_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        mlflow.log_metric("training accuracy", train_accuracy)
        dump(dt_classifier, "models/activity_classifier.joblib")
        mlflow.log_artifact("models/activity_classifier.joblib")
        if production_ready:
            mlflow.set_tag("production_ready", 1)
        else:
            mlflow.set_tag("production_candidate", 1)
        os.remove("models/activity_classifier.joblib")


if __name__ == "__main__":
    train()
