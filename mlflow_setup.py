import mlflow

from config import Config

if __name__ == "__main__":
    mlflow.set_tracking_uri(Config.TRACKING_URI)
    experiment_id = mlflow.create_experiment(
        Config.EXPERIMENT_NAME,
        artifact_location=f"s3://{Config.BUCKET_NAME}/{Config.EXPERIMENT_NAME}",
    )
    print(f"Your experiment_id is {experiment_id}. Copy it to config.py!")
