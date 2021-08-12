import joblib
import tensorflow as tf
from loguru import logger
from transformers import TFDistilBertForSequenceClassification


def load_model_loaders():
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    load_mechanisms = {
        "joblib": joblib.load,
        "keras": tf.keras.models.load_model,
        "transformers": TFDistilBertForSequenceClassification.from_pretrained
    }
    logger.info("Loaders defined")
    logger.info(load_mechanisms.keys())
    return load_mechanisms
