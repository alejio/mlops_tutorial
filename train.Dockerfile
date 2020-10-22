FROM python:3.8.5
WORKDIR /app
COPY requirements-train.txt ./requirements-train.txt
RUN pip3 install -r requirements-train.txt
COPY . .
ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV ARTIFACT_LOCATION='s3'
ENV PRODUCTION_READY='--no-production-ready'
# Need to provide build arg to set env variable from host env variable:
# https://vsupalov.com/docker-build-pass-environment-variables/
CMD python train.py $ARTIFACT_LOCATION $PRODUCTION_READY

# docker build --build-arg MLFLOW_TRACKING_URI -f train.Dockerfile -t mlops_tutorial_train .

# docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -it mlops_tutorial_train
