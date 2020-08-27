import mlflow
import click

TRACKING_URI = "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"

query = "tags.live='1'"
experiment_id = "0"


def fetch_live_model_id(tracking_uri, experiment_id, query):
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query).iloc[
        0
    ]
    print(runs["run_id"].values[0])


if __name__ == "__main__":
    fetch_live_model_id(TRACKING_URI, experiment_id, query)
