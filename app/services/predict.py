import os

from core.config import (LOCAL_MODEL_DIR, LOCAL_MODEL_NAME, MODEL_VERSION,
                         PROJECT_NAME)
from core.errors import ModelLoadException, PredictException
from loguru import logger


class MachineLearningModelHandlerScore(object):
    model = None

    @classmethod
    def predict(cls, input, load_wrapper=None, method="predict"):
        clf = cls.get_model(load_wrapper)
        if hasattr(clf, method):
            return getattr(clf, method)(input)
        raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper):
        model = None
        if LOCAL_MODEL_DIR.endswith("/"):
            path = f"{LOCAL_MODEL_DIR}{LOCAL_MODEL_NAME}"
        else:
            path = f"{LOCAL_MODEL_DIR}/{LOCAL_MODEL_NAME}"
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        model = load_wrapper(path)
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return model


class WANDBHandler(object):

    def download_latest_model():
        import wandb
        run = wandb.init(project=PROJECT_NAME, job_type="inference")
        model_at = run.use_artifact(LOCAL_MODEL_NAME.split(".")[
                                    0] + ":" + MODEL_VERSION)

        if LOCAL_MODEL_DIR.endswith("/"):
            path = f"{LOCAL_MODEL_DIR}{LOCAL_MODEL_NAME}"
        else:
            path = f"{LOCAL_MODEL_DIR}/{LOCAL_MODEL_NAME}"

        if os.path.exists(path):
            message = f"Machine learning model at {path} exists! Overwriting"
            logger.info(message)
            os.remove(path)

        model_at.download(LOCAL_MODEL_DIR)
        return None
