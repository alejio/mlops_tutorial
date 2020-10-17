import mlflow
import logging
from config import Config
import typer
from utils import get_mlflow_run


logging.basicConfig(level=Config.LOGGING)


def fetch_live_model_id(tracking_uri, experiment_id):
    run_id = get_mlflow_run(tracking_uri, experiment_id, Config.LIVE_TAG)
    logging.info(f"Live model run_id is: {run_id}")
    print(run_id)


if __name__ == "__main__":
    typer.run(fetch_live_model_id)
