'''
Top level orchestrator of the project. To be called from the CLI.
It comprises all the "routes" you may want to call
'''
from email import header
import numpy as np
import pandas as pd
import os
from ts_boilerplate.dataprep import get_Xi_yi, get_X_y, get_folds, train_test_split
from ts_boilerplate.model import get_model, fit_model, predict_output
from ts_boilerplate.metrics import mape, mae
from ts_boilerplate.params import CROSS_VAL, ROOT_DIR, TRAIN, DATA
from typing import Tuple, List
import matplotlib.pyplot as plt


def train(data: np.ndarray, print_metrics: bool = False):
    """
    Train the model in this package on one fold `data` containing the 2D-array of time-series for your problem
    Returns `metrics_test` associated with the training
    """
    pass  # YOUR CODE HERE


def cross_validate(data: np.ndarray, print_metrics: bool = False):
    """
    Cross-Validate the model in this package on`data`
    Returns `metrics_cv`: the list of test metrics at each fold
    """
    pass  # YOUR CODE HERE


def backtest(data: np.ndarray,
             stride: int = 1,
             start_ratio: float = 0.9,
             retrain: bool = True,
             retrain_every: int = 1,
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

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(ROOT_DIR, 'data','raw','data.csv')).to_numpy()
    try:
        train(data=data, print_metrics=True)
        # cross_validate(data=data, print_metrics=True)
        # backtest(data=data,
        #      stride = 1,
        #      start_ratio = 0.9,
        #      retrain = True,
        #      retrain_every=1,
        #      print_metrics=True,
        #      plot_metrics=True)
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
