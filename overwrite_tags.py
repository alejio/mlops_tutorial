import mlflow
import click

TRACKING_URI = "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"
experiment_id = "0"
query = "tags.production_candidate='1'"


@click.command()
@click.option("--baseline_run_id")
@click.option("--candidate_run_id")
def overwrite_tags(baseline_run_id, candidate_run_id):
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    print(f"Setting live tag to 0 for existing live model run ID {baseline_run_id}")
    client.set_tag(baseline_run_id, "live", 0)
    print(f"Setting live tag to 1 for candidate model run ID {candidate_run_id}")
    client.set_tag(candidate_run_id, "live", 1)
    run_ids = client.search_runs(experiment_id=experiment_id, query=query)["run_id"]
    print("Cleaning up production_candidate tags within experiment")
    for run_id in run_ids:
        print(f"Setting production_candidate tag to 0 for run ID {run_id}")
        client.set_tag(candidate_run_id, "production_candidate", 0)


if __name__ == "__main__":
    overwrite_tags()
