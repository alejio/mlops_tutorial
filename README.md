# mlops_tutorial

## Initial setup
Clone: `git clone git@github.com:alejio/mlops_tutorial.git`

Create virtual environment: `conda create --name mlops_tutorial python=3.8.5`

Activate virtual environment: `conda activate mlops_tutorial`

Install all dependencies: `pip install -r requirements.txt`

## 0Ops

In this section we will deploy the ML-powered application to the world without "Ops" of any kind.

TODO: create app0

### Milestone 1: Deploy app to heroku

[See original instructions](https://devcenter.heroku.com/articles/container-registry-and-runtime)

Steps:

0. [Install heroku cli](https://devcenter.heroku.com/articles/heroku-cli)
1. Log in to cli: `heroku login -i`
2. Log in to Container Registry: `heroku container:login`
3. Change 
3. Create app: `heroku-create`. Take a note of the name of the created app!
4. Build image and push to Container Registry: `heroku container: push web`
5. Release image to app: `heroku container: release web`
6. See app in browser: `heroku open`

## AlmostOps: Start getting more serious
TODO: create appAlmost

### Milestone 2: Load artefacts from S3

### Milestone 3: Enable CD with Github Actions

This enables continuous deployment for the app with Github Actions, triggered on master branch push.

- Create secrets: `HEROKU_API_KEY` and `HEROKU_APP_NAME`
- Rename `deploy_app.yolo` to `deploy_app.yml` in `.github/workflows/`
- Commit and push!

## MLOps:

Here we do a lot of things:

- Enable model provenance tracking with MLflow running on an EC2 instance
- Use S3 as a store for the ML artefacts
- Enable model updating through CI/CD flow, and improve auditability through PR workflow.

Some setup first:

### Create Github app

- Github Settings -> Developer settings -> New Github App
- Give it a name and the repo URL
- Generate and set a private key 
- Permissions: all repo permissions read-only, except for content references (none), read & write for Deployments, Pull Requests, Workflows. #TODO!
- Install app to your account and give access to the repository
- Add secrets for Github app: `APP_ID`, `APP_PEM`


### Create S3 bucket

- Create a public S3 bucket and a directory `data`
- Update the S3 bucket name in config.py
- Dump `train.csv` and `test.csv` under `data`
- Add AWS creds to secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### Configure and launch EC2 instance

- Make sure aws cli is set up
- Run Cloudformation script

### Set up MLlow

- #TODO! `python mlflow_setup.py`
- Create secrets `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`

## All set!
- # TODO

### Create Github secrets

- MLOPS_TUTORIAL_TOKEN??