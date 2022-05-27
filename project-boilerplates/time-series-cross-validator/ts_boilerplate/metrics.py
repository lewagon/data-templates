'''
Computes usefull Time Series metrics from (y_true, y_test)
'''

import numpy as np
from tensorflow import reduce_mean
from tensorflow.keras.metrics import mean_absolute_error, mean_absolute_percentage_error


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns Mean Absolute Error"""
    # $CHALLENGIFY_BEGIN
    return reduce_mean(mean_absolute_error(y_true, y_pred)).numpy()
    # $CHALLENGIFY_END

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns Mean Absolute Percentage Error"""
    # $CHALLENGIFY_BEGIN
    return reduce_mean(mean_absolute_percentage_error(y_true, y_pred)).numpy()
    # $CHALLENGIFY_END

def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns Mean Absolute Scaled Error (https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
    """
    pass


def play_trading_strategy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the array of relative portfolio values over the test period"""
    pass


def return_on_investment(played_trading_strategy: np.ndarray) -> float:
    """Returns the ROI of an investment strategy"""
    pass


def sharpe_ratio(played_trading_strategy: np.ndarray) -> float:
    """Returns the Sharpe Ratio (Return on Investment / Volatility) of an investment strategy"""
    pass
