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
    # $CHALLENGIFY_BEGIN
    data_train, data_test = train_test_split(data, **TRAIN)
    X_train, y_train = get_X_y(data_train, **TRAIN)
    X_test, y_test = get_X_y(data_test, **TRAIN)
    model = get_model(X_train, y_train)
    history = fit_model(model, X_train, y_train)
    y_pred = predict_output(model, X_test)
    metrics_test = mae(y_test, y_pred)
    if print_metrics:
        print("### Test Metric: ", metrics_test)
    return metrics_test
    # $CHALLENGIFY_END


def cross_validate(data: np.ndarray, print_metrics: bool = False):
    """
    Cross-Validate the model in this package on`data`
    Returns `metrics_cv`: the list of test metrics at each fold
    """
    # $CHALLENGIFY_BEGIN
    folds = get_folds(data, **CROSS_VAL)
    metrics_cv = []
    for fold in folds:
        metrics_fold = train(fold, print_metrics=print_metrics)
        metrics_cv.append(metrics_fold)

    if print_metrics:
        print(f"### CV metrics after {len(folds)} folds ### ")
        print(metrics_cv)
    return metrics_cv
    # $CHALLENGIFY_END


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
    # $CHALLENGIFY_BEGIN

    # Initialization
    start_timestep_0 = round(start_ratio * len(data))
    data_train_0 = data[:start_timestep_0, ...]
    X_train_tmp, y_train_tmp = get_X_y(data_train_0, **TRAIN)
    data_test_backtested = data[start_timestep_0:, ...]
    _, y_test = get_X_y(data_test_backtested, **TRAIN, shuffle=False)
    y_pred_backtested = []
    retrain_counter = 0
    timesteps_backtested_list = []
    for i in range(0, len(data_test_backtested), stride):
        start_timestep_i = start_timestep_0 + i
        data_train = data[:start_timestep_i, ...]
        data_test = data[start_timestep_i:, ...]
        X_train_tmp, y_train_tmp = get_X_y(data_train, **TRAIN)
        X_test_i, y_test_i = get_Xi_yi(first_index=0, data=data_test, **TRAIN)

        # At some point after sliding through time, we will reach the end of the test set
        if y_test_i.shape[0] < y_train_tmp.shape[1]:
            break

        model = get_model(X_train_tmp, y_train_tmp)

        # Retrain when required, with incremental learning (ie. starting from previous weights)
        if retrain and i % retrain_every == 0:
            retrain_counter += 1
            fit_model(model, X_train_tmp, y_train_tmp)

        y_pred_i = np.squeeze(predict_output(model, X_test_i[None, ...]))
        y_pred_backtested.append(y_pred_i)
        timesteps_backtested_list.append(i)

    y_pred_backtested = np.array(y_pred_backtested)
    y_test_backtested = y_test[timesteps_backtested_list]
    # Check that we compare apples to apples
    assert y_pred_backtested.shape == y_test_backtested.shape

    metrics_backtested = mae(y_pred_backtested, y_test_backtested)

    if print_metrics:
        print(
            f'### BACKETESTED METRICS BASED ON THE LAST {y_pred_backtested.shape[0]} TIMESTEPS AND WITH {retrain_counter} retrain operations'
        )
        print(mae(y_pred_backtested, y_test_backtested))
    if plot_metrics:
        # TODO: make it work for any dimension of y
        plt.plot(y_pred_backtested[:,0,0], label='historical forecasts')
        plt.plot(y_test_backtested[:,0,0], label='truth')
        plt.xlabel('timesteps number (0=beginning of backtest)')
        plt.legend()
        plt.show()

    return metrics_backtested
    # $CHALLENGIFY_END

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(ROOT_DIR, 'data','raw','data.csv')).to_numpy()
    try:
        train(data=data, print_metrics=True)
        cross_validate(data=data, print_metrics=True)
        backtest(data=data,
             stride = 1,
             start_ratio = 0.9,
             retrain = True,
             retrain_every=1,
             print_metrics=True,
             plot_metrics=True)
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
