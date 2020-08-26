import mlflow

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


def render_markdown(baseline_run, candidate_run):
    # header_row = [ms.bold('Model'), ms.bold('Accuracy')]
    # baseline_row = [ms.italics('Baseline'), baseline_run['metrics.training accuracy']]
    # candidate_row = [ms.italics('Candidate'), candidate_run['metrics.training accuracy']]
    # output_table = ms.table_from_rows([header_row, baseline_row])
    output_table = f"| **Model** | **Accuracy** |\n| ---------- | -------------- |\n| _Baseline_ | {round(baseline_run['metrics.training accuracy'], 4)} |\n| _Candidate_ | {round(candidate_run['metrics.training accuracy'], 4)} |"
    print(output_table)


if __name__ == "__main__":
    baseline_run, candidate_run = fetch_runs(
        TRACKING_URI, experiment_id, baseline_query, candidate_query
    )
    render_markdown(baseline_run, candidate_run)
