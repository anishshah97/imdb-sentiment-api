import joblib
import tensorflow as tf


def load_model_loaders():
    """
    load the models from disk
    and put them in a dictionary
    Returns:
        dict: loaded models
    """
    load_mechanisms = {
        "joblib": joblib.load,
        "keras": tf.keras.models.load_model
    }
    print("Loaders defined")
    print(load_mechanisms.keys())
    return load_mechanisms
