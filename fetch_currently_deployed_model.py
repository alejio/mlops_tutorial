import mlflow
import logging
from config import Config
import typer


logging.basicConfig(level=Config.LOGGING)


def fetch_live_model_id():
    mlflow.set_tracking_uri(Config.TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_ids=[Config.EXPERIMENT_ID],
        filter_string=f"tags.{Config.LIVE_TAG}='1'",
    ).iloc[0]
    logging.info(runs["run_id"])
    return runs["run_id"]


if __name__ == "__main__":
    typer.run(fetch_live_model_id)
