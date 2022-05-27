import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Reshape, Lambda, Input
from tensorflow.keras import Model
from ts_boilerplate.params import DATA, TRAIN

# TODO: Should we add here the preprocessing? into a class called "pipeline"?
# TODO: Should we refacto in a class ? Probably!


def get_model(X_train, y_train):
    """Instanciate, compile and and return the model of your choice"""
    pass  # YOUR CODE HERE


def fit_model(model, X_train, y_train, **kwargs):
    """Fit the `model` object, including preprocessing if needs be"""
    pass  # YOUR CODE HERE


def predict_output(model, X_test):
    """Return y_test. Include preprocessing if needs be"""
    pass  # YOUR CODE HERE
