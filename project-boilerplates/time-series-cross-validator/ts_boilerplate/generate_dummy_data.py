import numpy as np
from ts_boilerplate.params import CROSS_VAL, DATA, TRAIN
from typing import Tuple

def generate_data_monotonic_increase() -> np.ndarray:
    """Creates a monotonicly increasing time serie dataset for test purposes
    - shape is (DATA['length'], DATA['n_covariates] + DATA['n_targets']),
    - values are all equals to their respective integer index!

    e.g:
    data = array(
      [[  0.,   0.,   0.,   0.,   0.],
       [  1.,   1.,   1.,   1.,   1.],
       ...,
       [998., 998., 998., 998., 998.],
       [999., 999., 999., 999., 999.]]
    )

    """

    indexes = np.arange(0, DATA['length'])
    data = np.zeros((DATA['length'], DATA['n_covariates'] + DATA['n_targets'])) \
        + np.expand_dims(indexes, axis=1)
    return data

def generate_data_zeros_and_ones() -> np.ndarray:
    """Create a dummy data made of zeros for covariates, and ones for the targets
    e.g:
    data = array(
      [[1.,1.,0.,0.,0.],
       [1.,1.,0.,0.,0.],
       ...,
       [1.,1.,0.,0.,0.],
       [1.,1.,0.,0.,0.]]
    )
    """
    shape = (DATA['length'], DATA['n_covariates'] + DATA['n_targets'])
    data = np.zeros(shape)
    data[:, DATA["target_column_idx"]] = 1.
    return data

def generate_X_y_zeros_and_ones() -> Tuple[np.ndarray]:
    """Create a dummy (X,y) tuple made of zeros for covariates, and ones for the targets, just to check if model fit well"""
    length = round(DATA["length"] / TRAIN['stride'])

    shape_X = (length, TRAIN['input_length'], DATA['n_covariates']+DATA['n_targets'])
    X = np.zeros(shape_X)
    X[:, :, DATA["target_column_idx"]] = 1.

    shape_y = (length, TRAIN['output_length'], DATA['n_targets'])
    y = np.ones(shape_y)
    y = np.squeeze(y)

    return (X,y)
