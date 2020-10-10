# mlops_tutorial

## Scope

Application code and supporting material for the [Codemotion 2020 workshop](https://events.codemotion.com/conferences/online/2020/codemotion-online-tech-conference/workshops/) titled "From 0Ops to MLOps" held on 22/10.

The project focusses on incrementally improving the supporting MLOps infrastructure of a simple ML-powered application.

Starting from the most basic config (0Ops), we gradually "decorate" the app with standard CI/CD workflows to arrive at a setup that enables basic ML auditability, reproducibility, and collaboration.

In this project we take advantage of basic functionality of state-of-the-art tools such as:

- [Streamlit](https://www.streamlit.io/), for easily creating a frontend for our app
- [Heroku](https://www.heroku.com) for easily deploying our Streamlit app to the world
- [MLflow (Tracking)](https://mlflow.org/docs/latest/tracking.html) for enabling ML artefact logging and experiment tracking
- [AWS EC2](https://aws.amazon.com/ec2/) for hosting the MLflow server that logs details of our experiments
- [AWS S3](https://aws.amazon.com/s3/) for storing the trained ML artefacts used by our application
- [Github Actions](https://github.com/features/actions) for creating CI/CD workflows that combine everything together and achieving "MLOps"

### Key learnings

1. Established DevOps practices are not sufficient for ML and Data Science powered applications
2. Experiment tracking is a central cog to an MLOps workflow. Many solutions exist, but MLflow is a nice starting point
3. Github Actions provides lots of cool functionality for MLOps
4. The Machine Learning Engineer role is vital - but don't expect to spend a lot of time doing model training

## Initial setup

- [x] Clone: `git clone git@github.com:alejio/mlops_tutorial.git`

- [x] Create virtual environment: `conda create --name mlops_tutorial python=3.8.5`

- [x] Activate virtual environment: `conda activate mlops_tutorial`

- [x] Install all dependencies: `pip install -r requirements.txt`

## 0Ops

In this section we will deploy the ML-powered application to the world without "Ops" of any kind.

### Note: Streamlit app

For the purposes of this tutorial we will use a very simple application that uses an ML model to predict the sentiment of a user-provided movie review.

The application itself is a slightly modified version of the `galleries/sentiment_analyzer` example Streamlit app found here awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit).

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
4. Push to master and check if it's working!

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