import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from joblib import dump
import typer
import boto3
import logging
from config import Config
from typing import Dict

logging.basicConfig(level=Config.LOGGING)

# Simple ML model training script using locally persisted training data and


def load_data(train_csv: str, test_csv: str) -> Dict:
    """
    Loads local csv's into pandas dataframes and does basic
    preprocessing for ML.
    """
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


def train() -> None:
    """
    Trains an ML model using the local training data and writes the artifacts locally.
    """
    data_dict = load_data(f"data/{Config.TRAIN_CSV}", f"data/{Config.TEST_CSV}")

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
    dump(feature_engineering, f"{os.getcwd()}/models/feature_engineering.joblib")
    dump(classifier, f"{os.getcwd()}/models/classifier.joblib")
    logging.info("Done persisting models!")


if __name__ == "__main__":
    typer.run(train)
