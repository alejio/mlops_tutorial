from joblib import load
import numpy as np
import mlflow
import streamlit as st
import os.path
import boto3


@st.cache
def load_model():
    TRACKING_URI = (
        "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    query = "tags.live='1'"
    experiment_id = "0"
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["attributes.end_time DESC"],
    )
    run_id = runs["run_id"].values[0]
    bucket_name = "workshop-mlflow-artifacts"
    print(f"Bucket Name: {bucket_name}")
    artifact_path = f"{experiment_id}/{run_id}/artifacts/activity_classifier.joblib"
    print(f"Artifact path: {artifact_path}")
    s3 = boto3.client("s3")
    print("Download start")
    s3.download_file(bucket_name, artifact_path, "activity_classifier.joblib")
    print("Downloaded!")
    model = load("activity_classifier.joblib")
    return model


st.markdown("## Human activity predictor from smartphones!")

feature_a = st.number_input("tBodyAccMag-mean()")
feature_b = st.number_input("angle(X,gravityMean)")
feature_c = st.number_input("angle(Y,gravityMean)")


total_features = 561

int_features = [0 for i in range(total_features)]
int_features[200] = feature_a
int_features[558] = feature_b
int_features[559] = feature_c

final_features = [np.array(int_features)]


if st.button("Predict"):
    model = load_model()
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f"The predicted activity is {prediction[0]}")
