import mlflow
import logging
from config import Config
import typer


logging.basicConfig(level=Config.LOGGING)


def overwrite_tags(baseline_run_id: str, candidate_run_id: str):
    client = mlflow.tracking.MlflowClient(tracking_uri=Config.TRACKING_URI)
    logging.info(
        f"Setting live tag to 0 for existing live model run ID {baseline_run_id}"
    )
    client.set_tag(baseline_run_id, Config.LIVE_TAG, 0)
    logging.info(f"Setting live tag to 1 for candidate model run ID {candidate_run_id}")
    client.set_tag(candidate_run_id, Config.LIVE_TAG, 1)
    logging.info(
        f"Setting production_candidate tag to 0 for candidate model run ID {candidate_run_id}"
    )
    client.set_tag(candidate_run_id, Config.CANDIDATE_TAG, 0)


if __name__ == "__main__":
    typer.run(overwrite_tags)
