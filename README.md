# imdb-sentiment-api

> Midterm submission for Anish Shah 4B Midterm assignment.

In this assignment we will be showing an end to end CI/CD flow for the deployment of binary negative-positive sentiment models trained on the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

# Table of Contents

[Description](https://github.com/anishshah97/imdb-sentiment-api#description)

[Development Requirements](https://github.com/anishshah97/imdb-sentiment-api#development-requirements)

## Description

### Deployment Architecture

Using [Seldon Core](https://github.com/SeldonIO/seldon-core) and [MlFlow](https://www.mlflow.org/) as reference we made a configurable, extensible, and reusable endpoint architecture for ease of adaptation for a variety of models.

Our emphasis lies on making it easy to load and overwrite the predict functionality of any intended model based on some passed configuration parameters.

![alt text](https://camo.githubusercontent.com/ce96f712535ab67177d197e4324deb07fab094752cb9564951b863703392fdb3/68747470733a2f2f6c6c2d626c6f672e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f706963676f2f32303230303730393136323731372e706e67)

> We can see immediately the benefit of having the ability to refer to only one model server image and configuring the deployment run of that image vs. having a seperate image needed for each deployment

![alt text](https://camo.githubusercontent.com/8ec343c20d6858572c6ff87c7254437a84e7efa120624ef5d21dd8d0a617d3bd/68747470733a2f2f6c6c2d626c6f672e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f706963676f2f32303230303730393136353631312e706e67)

> Write a common model class to run your flavor of model if not already supported, pass in some parameters, read the benefit of a fully scalable and functional endpoint

![alt text](https://camo.githubusercontent.com/36fbb10f7a05cee6f51c73dfc1c7f8ac55361fa0b771d5222b94cff2a51c5dab/68747470733a2f2f6c6c2d626c6f672e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f706963676f2f32303230303730393137303231352e706e67)

> Governance and reproducibility at every level is key

![alt text](https://camo.githubusercontent.com/baca90841adb01750e4960b403854dc609ef6d31626604ace25231291bb1311b/68747470733a2f2f6c6c2d626c6f672e6f73732d636e2d68616e677a686f752e616c6979756e63732e636f6d2f706963676f2f32303230303730393138313830392e706e67)

> Real tim metric tracking allows us to easily account for data drift and also ensure the performance of our model and endpoint meet our expected standards.

- In our case instead of Prometheus or any of the other extensive logging tools offered in Seldon, we use WANDB to log inference job types to the provided account at the provided project name

![alt text](https://image.slidesharecdn.com/oreillyaideployingmlmodelsatscale-181012115501/95/seldon-deploying-models-at-scale-20-638.jpg?cb=1539345462)

> In the extended ecosystem Seldon Core situates itself at the deployment stages of the machine learing lifecycle

Our model deployment is an extremely simplified form of this methodology with a strong focus on integrating well within a reproducible CI/CD flow with inherent connection with [Weights & Biases](https://wandb.ai/).

As such, in its current state this repo is best forked and used as a templating tool to build out a working E2E flow utilizing CI/CD with `GCP` and model registry capabilties via `wandb`.

### CI/CD

Below we capture the CI/CD scenarios that we would expect with our model endpoints.

- In the `automated` build scenario, we capture any changes in the source code for the model server, build the new resultant docker image, push the image to the container registry, and then deploy via cloud run. This captures the CI component.

![alt text
](https://miro.medium.com/max/998/1*SQcTRfQ2Cqoq18yofRsvTQ.png)

> Automated builds based on changes in the `master` branch

- In the `scheduled` build scenario, to ensure that we pull the latest model from `wandb` we force the fastapi application to rebuild, which in turn queries the service for the latest recorded model. This ensures we are always serving the most up-to-date model at the endpoint.

![alt text](https://miro.medium.com/max/504/0*JR7aBMi66GFJlv5L)

> Scheduled builds on `master` to update the endpoint with the latest model

## Development Requirements

- Python3.9.2
- Pip
- Poetry (Python Package Manager)

### M.L Model Environment

```sh
LOCAL_MODEL_DIR = config("LOCAL_MODEL_DIR", default="./ml/model/")
LOCAL_MODEL_NAME = config("LOCAL_MODEL_NAME", default="model.pkl")
MODEL_VERSION = config("MODEL_VERSION", default="latest")
MODEL_LOADER = config("MODEL_LOADER", default="joblib")
WANDB_API_KEY=<API_KEY>
```

### M.L. Model Flavors

[Currently we only have added](app/core/model_loaders.py)

```sh
joblib.load
tf.keras.models.load_model
TFDistilBertForSequenceClassification.from_pretrained
```

### Update `core.events` in `main.py`

In `main.py` we reference a `startup` handler which we imported from `core.events` [(shown here)](app/core/events.py).
This runs on startup of the application.

On startup we use [Weights & Biases](https://wandb.ai/) to pull `LOCAL_MODEL_NAME` from their service and then use `MODEL_LOADER` to load the model before serving the application endpoints.

### Update `/predict`

To update your machine learning model, add your `load` and `method` [change here](app/api/routes/predictor.py) at `predictor.py`.

We adapted the predictor model loader based on `joblib` overwrote the predict function to better suit our `TFDistilBertForSequenceClassification` model.

## Installation

```sh
python -m venv venv
source venv/bin/activate
make install
```

## Runnning Localhost

`make run`

## Deploy app

`make deploy`

## Running Tests

`make test`

## Runnning Easter Egg

`make easter`

## Access Swagger Documentation

> <http://localhost:8080/docs>

## Access Redocs Documentation

> <http://localhost:8080/redoc>

## Project structure

Files related to application are in the `app` or `tests` directories.
Application parts are:

    app
    ├── api              - web related stuff.
    │   └── routes       - web routes.
    ├── core             - application configuration, startup events, logging.
    ├── models           - pydantic models for this application.
    ├── services         - logic that is not just crud related.
    └── main.py          - FastAPI application creation and configuration.
    │
    tests                  - pytest
