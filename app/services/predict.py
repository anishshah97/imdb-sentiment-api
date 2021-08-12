import os
import shutil

import tensorflow as tf
import wandb
from core.config import (LOCAL_MODEL_DIR, LOCAL_MODEL_NAME, MODEL_VERSION,
                         PROJECT_NAME)
from core.errors import ModelLoadException, PredictException
from core.model_loaders import load_model_loaders
from loguru import logger
from transformers import (DistilBertTokenizerFast,
                          TFDistilBertForSequenceClassification)


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


class BERTModelHandler(MachineLearningModelHandlerScore):
    # model = TFDistilBertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased")
    model = None
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased")

    @classmethod
    def predict(cls, text):
        model = cls.get_model(load_model_loaders()["transformers"])
        tokenizer = cls.tokenizer

        with wandb.init(project=PROJECT_NAME, job_type="inference") as run:
            tf_batch = tokenizer(
                text, max_length=128, padding=True, truncation=True, return_tensors='tf')
            tf_outputs = model(tf_batch)
            tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
            labels = ['Negative', 'Positive']
            label = tf.argmax(tf_predictions, axis=1)
            label = label.numpy()
            confidences = tf_predictions[0]
            negative_confidence = confidences[0]
            positive_confidence = confidences[1]
            resp = {'text': text, 'sentiment': labels[label[0]], 'confidences': {
                'negative': float(negative_confidence), 'positive': float(positive_confidence)}}

            wandb_log = {
                "negative": resp["confidences"]["negative"],
                "positive": resp["confidences"]["positive"]
            }

            run.log(wandb_log)

        return resp

    # @classmethod
    # def get_model(cls, **kwargs):
    #     weights_path = cls.get_weights_path()
    #     logger.debug(weights_path)
    #     cls.model.load_weights(weights_path)
    #     return cls.model

    # @staticmethod
    # def get_weights_path(**kwargs):
    #     if LOCAL_MODEL_DIR.endswith("/"):
    #         path = f"{LOCAL_MODEL_DIR}{LOCAL_MODEL_NAME}"
    #     else:
    #         path = f"{LOCAL_MODEL_DIR}/{LOCAL_MODEL_NAME}"
    #     if not os.path.exists(path):
    #         message = f"Machine learning model at {path} not exists!"
    #         logger.error(message)
    #         raise FileNotFoundError(message)
    #     return path+"/weights.h5"


class WANDBHandler(object):

    def download_latest_model():
        with wandb.init(project=PROJECT_NAME, job_type="download") as run:
            model_at = run.use_artifact(LOCAL_MODEL_NAME.split(".")[
                                        0] + ":" + MODEL_VERSION)

            if LOCAL_MODEL_DIR.endswith("/"):
                path = f"{LOCAL_MODEL_DIR}{LOCAL_MODEL_NAME}"
            else:
                path = f"{LOCAL_MODEL_DIR}/{LOCAL_MODEL_NAME}"

            if os.path.exists(path):
                message = f"Machine learning model at {path} exists! Overwriting"
                logger.info(message)
                shutil.rmtree(path)

            model_at.download(path)
        return None
