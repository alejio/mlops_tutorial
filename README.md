# mlops_tutorial

## Initial setup

> Clone: `git clone git@github.com:alejio/mlops_tutorial.git`
> Create virtual environment: `conda create --name mlops_tutorial python=3.8.5`
> Activate virtual environment: `conda activate mlops_tutorial`
> Install all dependencies: `pip install -r requirements.txt`

## 0Ops

In this section we will deploy the ML-powered application to the world without "Ops" of any kind.

### Milestone 1: Deploy app to heroku

[See original instructions](https://devcenter.heroku.com/articles/container-registry-and-runtime)

Steps:

0. [Install heroku cli](https://devcenter.heroku.com/articles/heroku-cli)
1. Log in to cli: `heroku login -i`
2. Log in to Container Registry: `heroku container:login`
3. Create app: `heroku-create`. Take a note of the name of the created app!
4. Build image and push to Container Registry: `heroku container: push web`
5. Release image to app: `heroku container: release web`
6. See app in browser: `heroku open`

## AlmostOps: Start getting more serious

In this section, we will enable loading of artefacts from S3, and enable Continuous Deployment of the application, using a Github Action triggered upon any push to master.

### Milestone 2: Load artefacts from S3

- Create a public S3 bucket and a directory `data`
- Update the S3 bucket name in config.py
- Dump `train.csv` and `test.csv` under `data`
- Add AWS creds to secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### Milestone 3: Enable CD with Github Actions

This enables continuous deployment for the app with Github Actions, triggered on master branch push.

- Create secrets: `HEROKU_API_KEY` and `HEROKU_APP_NAME`
- Rename `deploy_app.yolo` to `deploy_app.yml` in `.github/workflows/`
- Commit and push!

## MLOps-ready

In the final section, we enable model training through a CI/CD Github Actions flow, and use an MLflow Tracking server deployed on an EC2 instance to record all information relating to trained ML models.

Finally, we use Github Actions to create a better pull request workflow for updating trained models.

### Milestone 4: Set up MLflow

1. Launch EC2 instance
2. Install MLflow
3. Configure nginx
4. Create Github secrets `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`

(TODO: figure out what I have to do with IAM roles)
(TODO: for EC2 consider using a CloudFormation script)
(TODO: consider using a `python mlflow_setup.py`)

### Milestone 5: MLflow instrumentation of application

1. "Decorate" training script with MLflow logging functionality
2. Run training and check MLflow and S3 for new artefacts
3. Update `app.py` to pick the right model S3 artefact using queries to MLflow server  

### Milestone 6: Enable Github Actions for PR workflow

Some setup first:

- Github Settings -> Developer settings -> New Github App
- Give it a name and the repo URL
- Generate and set a private key 
- Permissions: all repo permissions read-only, except for content references (none), read & write for Deployments, Pull Requests, Workflows. #TODO!
- Install app to your account and give access to the repository
- Add secrets for Github app: `APP_ID`, `APP_PEM`

## All set

TODO: MLOPS_TUTORIAL_TOKEN??