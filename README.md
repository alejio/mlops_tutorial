# mlops_tutorial

## Scope

Application code and supporting material for the [Codemotion 2020 workshop](https://events.codemotion.com/conferences/online/2020/codemotion-online-tech-conference/workshops/) titled "From 0Ops to MLOps" held on 22/10.

The project focusses on incrementally improving the supporting MLOps infrastructure of a simple ML-powered application.

Starting from the most basic config (0Ops), we gradually "decorate" the app over a series of **Milestones** with standard CI/CD workflows to arrive at a setup that enables basic ML auditability, reproducibility, and collaboration.

In this project we take advantage of basic functionality of state-of-the-art tools such as:

- [Streamlit](https://www.streamlit.io/), for easily creating a frontend for our app
- [Heroku](https://www.heroku.com) for easily deploying our Streamlit app to the world
- [MLflow (Tracking)](https://mlflow.org/docs/latest/tracking.html) for enabling ML artifact logging and experiment tracking
- [AWS EC2](https://aws.amazon.com/ec2/) for hosting the MLflow server that logs details of our experiments
- [AWS S3](https://aws.amazon.com/s3/) for storing the trained ML artifacts used by our application
- [Github Actions](https://github.com/features/actions) for creating CI/CD workflows that combine everything together and achieving "MLOps"

> TODO: Link to app

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

In this section, we will enable loading of artifacts from S3, and enable Continuous Deployment of the application, using a Github Action triggered upon any push to master.

### Milestone 2: Load artifacts from S3

- Create a public S3 bucket and a directory `data`
- Update the S3 bucket name in config.py
- Dump `train.csv` and `test.csv` under `data`
- Add AWS creds to secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### Milestone 3: Enable CD with Github Actions

This enables continuous deployment for the app with Github Actions, triggered on master branch push.

- Create secrets: `HEROKU_API_KEY` and `HEROKU_APP_NAME`
- Rename `deploy_app.disabled` to `deploy_app.yml` in `.github/workflows/`
- Commit and push!

## MLOps-ready

In the final section, we enable model training through a CI/CD Github Actions flow, and use an MLflow Tracking server deployed on an EC2 instance to record all information relating to trained ML models.

Finally, we use Github Actions to create a better pull request workflow for updating trained models.

### Milestone 4: Set up MLflow
1. Launch EC2 instance
    - Create IAM role EC2 with S3 access
    - Amazon Linux AMI 2018.03.0 (HVM), SSD Volume Type - ami-0765d48d7e15beb93
    - In configure instance specify the IAM role you created
    - In security group create:
        - HTTP: source 0.0.0.0/0, ::/0
        - SSH: source 0.0.0.0/0
    - Create key/value pair and save pem file to the repo directory (it is gitignored)
    - Run `chmod 400 <your_key>.pem`
    
2. Install MLflow on EC2
    - From [link](https://medium.com/@alexanderneshitov/how-to-run-an-mlflow-tracking-server-on-aws-ec2-d7afd0ac8008)
    - From repo directory ssh into EC2 instance: `ssh -i "<your_key>.pem" ec2-user@ec2<your-instance>`
    - Install MLflow: `sudo pip install mlflow`
    - Downgrade dateutil (LOL): `sudo pip install -U python-dateutil==2.6.1`
    - Install boto3: `sudo pip install boto3`
    
3. Configure nginx
    - Install nginx: `sudo yum install nginx`
    - Start nginx: `sudo service nginx start`
    - Install httpd tools to allow password protection: `sudo yum install httpd-tools`
    - Create password for user testuser: `sudo htpasswd -c /etc/nginx/.htpasswd testuser`
    - Enable global read/write permissions to nginx directory: `sudo chmod 777 /etc/nginx`
    - Delete nginx.conf so we replace it with a modified one: `rm /etc/nginx/nginx.conf`
    - Open new terminal window and upload the nginx.conf file in this repo to EC2: `scp -i <your_key>.pem nginx.conf ec2-user@ec2<your-instance>:/etc/nginx/`
    - Reload nginx: `sudo service nginx reload`
    
4. Run MLflow server
    - Start the server: `mlflow server --default-artifact-root s3://<your-s3-bucket> --host 0.0.0.0`
    - Check it out! Open browser and go to your instance.

### Milestone 5: Instrument application with MLflow

1. Configure application to communicate with MLflow server
    - `export MLFLOW_TRACKING_URI=http://testuser:<your_password>@ec2-<your_instance>`
    - Run `mlflow_setup.py` and note your experiment_id
    - Copy your experiment_id to `config.py`
    - Run an initial experiment to create your first live model: `python train.py s3_mlflow --production-ready`
    - Check your MLflow server (refresh page) and be excited!
2. Run the streamlit app with MLflow artefacts locally: `streamlit run app.py s3_mlflow`
3. Create Github secrets
    - `MLFLOW_TRACKING_URI`: use format "http://testuser:<your_ec2_password>@ec2-<your_instance>"
4. In Dockerfile set `ARTIFACT_LOCATION=s3_mlflow`
5. Push to master, wait for the deployment and check out app.

### Milestone 6: Embed MLOps to pull requests

Some setup first:

1. Create a Github app for enabling a specific Actions step:
    - Github Settings -> Developer settings -> New Github App
    - Give it a name and the repo URL
    - Generate and set a private key 
    - Set permissions: all repo permissions read-only, except for content references (none), read & write for Deployments, Pull Requests, Workflows. #TODO!
    - Install app to your account and give access to the repository
    - Add secrets for Github app: `APP_ID`, `APP_PEM`
    
2. Enable Actions
    - Rename `deploy_app.disabled` to `deploy_app.yml`
    - Rename `evaluate.disabled` to `evaluate.yml`
    
3. See the workflow in action!
    - Create a new branch: `git checkout -b test-ml-pr`
    - Update a model config param
    - Create a PR
    - Enter `/evaluate` in the PR chat and see magic happening
    - Enter `/deploy-candidate` in the PR chat and wait for magic to happen
    - Merge PR to redeploy app
    - Wait for the action to complete and checkout the app

> TODO: rename master branch to main

> TODO: MLOPS_TUTORIAL_TOKEN??