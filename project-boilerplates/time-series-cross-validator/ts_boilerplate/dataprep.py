"""Prepare Data so as to be used in a Pipelined ML model"""

import numpy as np
from ts_boilerplate.params import DATA
from typing import Tuple, List
import numpy as np


def load_data(data_path: str) -> np.ndarray:
    """Load data from `data_path` into to memory
    Returns a 2D array with (axis 0) representing timesteps, and (axis 1) columns containing tagets and covariates
    ref: https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true
    """
    # YOUR_CODE_HERE
    pass


def clean_data(data: np.ndarray) -> np.ndarray:
    """Clean data without creating data leakage:
        - make sure there is no NaN between any timestep
        - etc...
    """
    # YOUR_CODE_HERE
    pass


def get_X_y(
    data: np.ndarray,
    input_length: int,
    output_length: int,
    horizon: int,
    stride: int,
    shuffle=True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use `data`, a 2D-array with axis=0 as timesteps, and axis=1 as (tagets+covariates columns)

    Returns a Tuple (X,y) of two ndarrays :
        X.shape = (n_samples, input_length, n_covariates)
        y.shape =
            (n_samples, output_length, n_targets) if all 3-dimensions are of size > 1
            (n_samples, output_length) if n_targets == 1
            (n_samples, n_targets) if output_length == 1
            (n_samples, ) if both n_targets and lenghts == 1

    ❗️ Raise error if data contains NaN
    ❗️ Make sure to shuffle the pairs in unison if `shuffle=True` for idd purpose
    ❗️ Don't ditch past values of your target time-series in your features - they are very useful features!
    👉 illustration: https://raw.githubusercontent.com/lewagon/data-images/master/DL/rnn-1.png

    [💡 Hints ] You can use a sliding method
        - Reading `data` in ascending order
        - `stride` timestamps after another
    Feel free to use another approach, for example random sampling without replacement

    """
    # $CHALLENGIFY_BEGIN
    assert np.isnan(data).sum() == 0

    X = []
    y = []

    for i in range(0, len(data), stride):
        Xi, yi = get_Xi_yi(first_index=i,
                           data=data,
                           horizon=horizon,
                           input_length=input_length,
                           output_length=output_length)
        # Exit loop as soon as we reach the end of the dataset
        if len(yi) < output_length:
            break
        X.append(Xi)
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    y = np.squeeze(y)
    if shuffle:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    return X, y
    # $CHALLENGIFY_END


# $DELETE_BEGIN
def get_Xi_yi(first_index,
              data,
              horizon,
              input_length,
              output_length,
              **kwargs):
    X_start = first_index
    X_last = X_start + input_length
    y_start = X_last + horizon - 1
    y_last = y_start + output_length

    Xi = data[X_start:X_last]
    yi = data[y_start:y_last, DATA['target_column_idx']]
    return (Xi, yi)
# $DELETE_END


def get_folds(data: np.ndarray,
              fold_length: int,
              fold_stride: int,
              **kwargs) -> List[np.ndarray]:
    """Slide through `data` time-series (2D array) to create folds of equal `fold_length`, using `fold_stride` between each fold
    Returns a list of folds, each as a 2D-array time series
    """
    # $CHALLENGIFY_BEGIN
    folds = []
    for i in range(0, len(data), fold_stride):
        # Exit loop as soon as last fold value would exceed last data value
        if (i + fold_length) > len(data):
            break
        fold = data[i:i + fold_length, :]
        folds.append(fold)
    return folds
    # $CHALLENGIFY_END


def train_test_split(data: np.ndarray,
                     train_test_ratio: float,
                     input_length: int,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a train and test 2D-arrays, that will not create any data leaks when sampling (X, y) from them
    Inspired from "https://raw.githubusercontent.com/lewagon/data-images/master/DL/rnn-3.png"
    """
    # $CHALLENGIFY_BEGIN
    last_train_idx = round(train_test_ratio * len(data))
    data_train = data[0:last_train_idx, :]

    # [here is the key to no data leak]
    # The last idx of the first X_test must be equal to the last idx of the last y_train.
    # Its equal to day n°10 in the picture rnn-3.png
    first_test_idx = last_train_idx - input_length
    data_test = data[first_test_idx:, :]

    return (data_train, data_test)
    # $CHALLENGIFY_END
