import mlflow
import click

TRACKING_URI = "http://testuser:test@ec2-3-9-174-162.eu-west-2.compute.amazonaws.com"

baseline_query = "tags.production_ready='1'"
candidate_query = "tags.production_candidate='1'"
experiment_id = "0"


def fetch_runs(tracking_uri, experiment_id, baseline_query, candidate_query):
    mlflow.set_tracking_uri(TRACKING_URI)
    baseline_run = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string=baseline_query
    ).iloc[0]
    candidate_run = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string=candidate_query
    ).iloc[0]
    return baseline_run, candidate_run


@click.command()
@click.option("--run")
@click.option("--attribute")
def emit_variable(run, attribute):
    baseline_run, candidate_run = fetch_runs(
        TRACKING_URI, experiment_id, baseline_query, candidate_query
    )
    if run == "baseline":
        click.echo(f"{baseline_run[attribute]}")
    else:
        click.echo(f"{candidate_run[attribute]}")


if __name__ == "__main__":
    emit_variable()
