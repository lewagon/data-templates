'''
Top level orchestrator of the project. To be called from the CLI.
It comprises all the "routes" you may want to call
'''
import numpy as np
from ts_boilerplate.dataprep import get_Xi_yi, get_X_y, get_folds, train_test_split
from ts_boilerplate.model import get_model, fit_model, predict_output
from ts_boilerplate.metrics import mape
from ts_boilerplate.params import CROSS_VAL, TRAIN, DATA
from typing import Tuple, List
import matplotlib.pyplot as plt


def train(data: np.ndarray, print_metrics: bool = False, save_metrics: bool = False):
    """
    Train the model in this package on one fold `data` containing the 2D-array of time-series for your problem
    Returns `metrics_test` associated with the training
    """
    pass  # YOUR CODE HERE


def cross_validate(data: np.ndarray, print_metrics: bool = False, save_metrics: bool = False):
    """
    Cross-Validate the model in this package on`data`
    Returns `metrics_cv`: the list of test metrics at each fold
    """
    pass  # YOUR CODE HERE


def backtest(data: np.ndarray,
             stride: int = 10,
             start_ratio: float = 0.8,
             retrain: bool = True,
             retrain_every: int = 30,
             print_metrics=False,
             plot_metrics=False):
    """Returns historical forecasts for the entire dataset
    - by training model up to `start_ratio` of the dataset
    - then predicting next values using the model in this package (only predict the last time-steps if `predict_only_last_value` is True)
    - then moving `stride` timesteps ahead
    - then retraining the model if `retrain` is True and if we moved `retrain_every` timesteps since last training
    - then predicting next values again

    Return:
    - all historical predictions as 2D-array time-series of shape ((1-start_ratio)*len(data), n_targets)/stride
    - Compute the 'mean-MAPE' per forecast horizon
    - Print historical predictions if you want a visual check

    see https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel.historical_forecasts
    """
    pass  # YOUR CODE HERE
