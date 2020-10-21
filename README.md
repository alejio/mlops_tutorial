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


## Initial setup

- [x] Fork this repo
- [x] Clone your forked repo: `git clone git@github.com:<your-gh-name>/mlops_tutorial`
- [x] Create virtual environment: `conda create --name mlops_tutorial python=3.8.5`
- [x] Activate virtual environment: `conda activate mlops_tutorial`
- [x] Install all dependencies: `pip install -r requirements.txt`
- [x] Make sure all files under `.github/workflow` have file extension `.disabled`. if there is a `*.yml`, rename it to `*.diabled`
- [x] Make sure ARTIFACT_LOCATION='local' in `Dockerfile` and `train.Dockerfile`

## Overview

- My app deployed on Heroku: https://polar-oasis-38285.herokuapp.com/
- The Streamlit app is `app.py`. The app has been dockerised with `Dockerfile`
- The ML model training script producing artefacts is `train.py`. It has also been dockerised with `train.Dockerfile`
- The `ARTIFACT_LOCATION` parameter in both app and training Dockerfiles controls the various stages of the workshop:
    - 0Ops stage: `ARTIFACT_LOCATION='local`
    - AlmostOps stage: `ARTIFACT_LOCATION='s3'`
    - MLOps stage: `ARTIFACT_LOCATION='s3_mlflow`
 - S3 bucket: https://s3.console.aws.amazon.com/s3/buckets/workshop-mlflow-artifacts/?region=eu-west-2&tab=overview
 - MLflow server: http://ec2-18-134-150-82.eu-west-2.compute.amazonaws.com/

## 0Ops

In this section we will 
- Deploy the ML-powered application to the world without "Ops" of any kind.

### Note: Streamlit app

For the purposes of this tutorial we will use a very simple application that uses an ML model to predict the sentiment of a user-provided movie review.

The application itself is a slightly modified version of the `galleries/sentiment_analyzer` example Streamlit app found here [awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit).


### Milestone 1: Deploy our app to heroku

[Make sure you have a Heroku account and installed cli](https://devcenter.heroku.com/articles/heroku-cli).

First, we will specify to the app we are running in *0Ops* mode:
 - Set `ENV ARTIFACT_LOCATION='local'` in `Dockerfile` and `train.Dockerfile`

Secondly, we will use heroku cli to [build and deploy the application Docker container to Heroku](https://devcenter.heroku.com/articles/container-registry-and-runtime)
), and to the world!

- Steps:

1. Log in to cli: `heroku login -i`
2. Log in to Container Registry: `heroku container:login`
3. Create app: `heroku create`. Take a note of the name of the created app!
4. Add secrets to your Github repo (repo/settings/secrets). We will need this later.
- `HEROKU_EMAIL`: the email you use on Heroku
- `HEROKU_APP_NAME`: the output of step 3: e.g. polar-oasis-12478
- `HEROKU_API_KEY`: get from [here](https://dashboard.heroku.com/account)
5. Build container and push to Heroku Container Registry. This will take a while!: `heroku container:push web`
6. Release uploaded container to app: `heroku container:release web`
7. See public app in browser: `heroku open`

## AlmostOps: Start getting more serious

In this section, we will 
- Enable loading of artifacts from S3
- Enable Continuous Deployment of the application, using a Github Action triggered upon any push to master.

For the purposes of the workshop we will use my S3 bucket, in order to mitigate issues with setting up.

Some setup first!

You need permissions to read and write to the workshop S3 bucket:
- Create a file named `.env`
- Add the following lines:
    - `export AWS_ACCESS_KEY_ID=<the key I send you on Discord>`
    - `export AWS_SECRET_ACCESS_KEY=<the key I send you on Discord>`
    - Run `source .env` in your terminal
- Also, add the above AWS credentials as secrets in Github (repo/settings/secrets), which we will need later. 
 
### Milestone 2: Load artifacts from S3

In this stage, we will instruct the app to load artefacts from S3, rather than its local environment.

The two changes are: 
1. Make training script write artefacts to dedicated subdirectory for each participant in the workshop's [S3 bucket](https://s3.console.aws.amazon.com/s3/buckets/workshop-mlflow-artifacts/?region=eu-west-2&tab=overview). 
    - In `config.py` set the `Config` class attribute `USER` to something unique; e.g. your Github handle
    - In `train.Dockerfile` set `ARTIFACT_LOCATION='s3`
    TODO: pass creds directly
    - Build training container: `docker build -f train.Dockerfile -t mlops_tutorial_train .`
    - Run training container: `docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -it mlops_tutorial_train`
    - Alternatively, do: `python train.py s3`
2. Instruct app to load artefacts from S3, rather than the local environment
    - In `Dockerfile` set `ARTIFACT_LOCATION=s3`
    

### Milestone 3: Enable Continuous Deployment with Github Actions

This enables continuous deployment for the app with Github Actions, triggered on master branch push.

- Rename `deploy_app.disabled` to `deploy_app.yml` in `.github/workflows/`
- Change the email field to your own
- Commit all your changes to git and push to master
    - `git add .`
    - `git commit -m 'Milestone 3'`
    - `git push`

The Github Action knows to deploy the right Heroku app, because of the secrets we added to Github `HEROKU_APP_NAME` and `HEROKU_API_KEY` earlier.

Now, the app will be automatically redeployed whenever you modify the master branch!

## MLOps

In the final section, we 
- Introduce experiment tracking with MLflow Tracking server, deployed on an EC2 instance
- Enable model training to take place automatically within a CI/CD Github Actions flow, rather than manually 
- Finally, we use Github Actions to create a cool pull request workflow for updating the models!

### Milestone 4: Instrument application with MLflow

Here, we will leverage simple "decorations" in the application and training jobs to achieve MLflow instrumentation.

1. Configure application to communicate with MLflow server
    - Add to .env file a new variable: `MLFLOW_TRACKING_URI=http://testuser:test@ec2-18-134-150-82.eu-west-2.compute.amazonaws.com/`
    - Run `source .env` in terminal
    - Also, add `MLFLOW_TRACKING_URI` as a secret in Github (repo/settings/secrets) 
    - Run `mlflow_setup.py`, note your experiment_id and overwrite the existing value in `config.py`
    - In `Dockerfile` set `ARTIFACT_LOCATION=s3_mlflow`
    - In `train.Dockerfile` set `ARTIFACT_LOCATION=s3_mlflow`

2. Run a training job to register your model with MLflow:
    - `python train.py s3_mlflow --production-ready`
    - Go to the [MLflow server](http://ec2-18-134-150-82.eu-west-2.compute.amazonaws.com/) and be excited!

3. Commit and push to master, wait for the automated deployment and check out app!
    - `git add.`
    - `git commit -m "Milestone 4"`
    - `git push`

### Milestone 5: Enable `evaluate` Github Action

This Github Action has been set to be triggered from a PR comment, but we could also have chosen it to be triggered by push to master.

We will see it in action in the next milestone.

For now, all we have to do is to:
-  Rename `evaluate.disabled` to `evaluate.yml`
- Commit and push to master:
    - `git add .`
    - `git commit -m "Milestone 5`
    - `git push`


### Milestone 6: Embed MLOps to pull requests

Now we get to see the workflow in action!!
- Create a new branch: `git checkout -b test-ml-pr`
- Update any model config param in `train.py`; e.g. `alpha=0.9`
- Open a pull request against this branch
- Enter `/evaluate` in the PR chat and see magic starting to happen
- Check out the model results
- Enter `/deploy-candidate` in the PR chat and wait for more magic to happen
- Now, merge the PR to redeploy Heroku app
- Wait for the action to complete and checkout the app!

## Appendix

### Set up MLflow server
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
