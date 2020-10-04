from joblib import load
import numpy as np
import mlflow
import streamlit as st
import os.path
import boto3
import logging

logging.basicConfig(level=logging.info)


@st.cache
def load_artefacts():
    TRACKING_URI = (
        "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    query = "tags.live='1'"
    experiment_id = "2"
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["attributes.end_time DESC"],
    )
    run_id = runs["run_id"].values[0]
    bucket_name = "workshop-mlflow-artifacts"
    logging.info(f"Bucket Name: {bucket_name}")
    feature_engineering_path = (
        f"predict_sentiment/{run_id}/artifacts/feature_engineering.joblib"
    )
    classifier_path = f"predict_sentiment/{run_id}/artifacts/classifier.joblib"
    logging.info(f"Artifact path: {classifier_path}")
    s3 = boto3.client("s3")
    logging.info("Download start")
    s3.download_file(
        bucket_name, feature_engineering_path, "feature_engineering.joblib"
    )
    s3.download_file(bucket_name, classifier_path, "classifier.joblib")
    logging.info("Downloaded!")
    feature_engineering = load("feature_engineering.joblib")
    classifier = load("classifier.joblib")
    return feature_engineering, classifier


st.title("Predicting movie review sentiment")
st.info(
    "Based on an example in [awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit) "
    "by [Marc Skov Madsen](https://github.com/MarcSkovMadsen), "
    "who took it from [Paras Patidar](https://github.com/patidarparas13/Sentiment-Analyzer-Tool). \n\n"
    "Cheers both!\n\n"
)
st.write(
    "The algorithm is trained on a collection of movie reviews and you can test it below."
)


st.subheader("Load model artefacts")
with st.spinner("Loading.."):
    feature_engineering, classifier = load_artefacts()
    st.info("Artefacts loaded successfully!")


st.title("Feed the hungry model with your review!")
write_here = "Write Here..."
review = st.text_input("Enter a review for classification by the algorithm", write_here)
if st.button("Predict Sentiment"):
    y = feature_engineering.transform([review])
    prediction = classifier.predict(y)
    probability = np.round(np.amax(classifier.predict_proba(y)), 2)

    def convert(prediction: int) -> str:
        return "Positive" if prediction == 1 else "Negative"

    if review != write_here:
        st.success(
            f"*Sentiment prediction*: **{convert(prediction).upper()}** with probability {100 * probability}%"
        )
        st.balloons()
    else:
        st.error("You need to input a review for classification!")
else:
    st.info(
        "**Enter a review** above and **press the button** to predict the sentiment."
    )
