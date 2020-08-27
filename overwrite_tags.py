import mlflow
import click

TRACKING_URI = "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"


@click.command()
@click.option("--baseline_run_id")
@click.option("--candidate_run_id")
def overwrite_tags(baseline_run_id, candidate_run_id):
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    client.set_tag(baseline_run_id, "live", 0)
    client.set_tag(candidate_run_id, "live", 1)
    client.set_tag(candidate_run_id, "production_candidate", 0)


if __name__ == "__main__":
    overwrite_tags()
