from typing import Callable

import joblib
from fastapi import FastAPI

from core.config import MODEL_LOADER
from core.model_loaders import load_model_loaders


def preload_model():
    """
    In order to load model on memory to each worker
    """
    # from services.predict import MachineLearningModelHandlerScore as model
    from services.predict import BERTModelHandler as model

    # TODO: Fix this so we can more easily use env variables to pass in the load mechanism
    model.get_model()


def download_latest_wandb_model():
    """
    In order to download model to load on memory for each worker
    """
    from services.predict import WANDBHandler

    WANDBHandler.download_latest_model()


def create_wandb_download_and_preload_handler(app: FastAPI) -> Callable:
    def model_app() -> None:
        download_latest_wandb_model()
        preload_model()

    return model_app
