# mlops_tutorial

## Clone repo

- `git clone .. `
- #TODO! `pip install -r requirements_full.txt`

## 0Ops:

### Deploy to heroku

Follow instructions https://devcenter.heroku.com/articles/container-registry-and-runtime

- Log in to cli: `heroku login -i`
- Log in to Container Registry: `heroku container:login`
- Create app: `heroku-create`. Note the name!
- Build image and push to Container Registry: `heroku container: push web`
- Release image to app: `heroku container: release web`
- Optionally open app in browser: `heroku open`

### Create Deploy action

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