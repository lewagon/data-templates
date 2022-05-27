import pytest
from ts_boilerplate.params import TRAIN, DATA
from ts_boilerplate.main import get_X_y
from ts_boilerplate.main import train_test_split
import numpy as np
import math


# These tests make use of the fixture `data_monotonic_increase` stored in tests/conftest.py (pytest magic under the hood)
def test_get_X_y_returns_correct_shapes(data_monotonic_increase):
    """Test that X and y have correct shape (excluding sample size), as per project setup as defined in `params.py`
    """
    X, y = get_X_y(data_monotonic_increase, **TRAIN)

    # Check that X and y have the correct lengths (in time) and depth (in number of covariates)
    assert X.ndim == 3
    assert X.shape[1] == TRAIN['input_length']
    assert X.shape[2] == DATA['n_covariates'] + DATA[
        'n_targets'], "Did you forget to include your past targets-values as features ?"

    y_should_be_3D = TRAIN['output_length'] > 1 and DATA["n_targets"] > 1
    y_should_be_1D = TRAIN['output_length'] == 1 and DATA["n_targets"] == 1
    if y_should_be_3D:
        assert y.ndim == 3
        assert y.shape[1] == TRAIN['output_length']
        assert y.shape[2] == DATA['n_targets']
    elif y_should_be_1D:
        assert y.ndim == 1
    else:
        assert y.ndim == 2
        assert y.shape[1] == TRAIN['output_length'] if DATA['n_targets'] == 1 else DATA['n_targets']


@pytest.mark.optional
@pytest.mark.skipif(TRAIN['stride'] == None, reason="Optional test only applicable if sliding method is used to get_X_y")
def test_optional_get_X_y_returns_optimal_sample_size(data_monotonic_increase):
    """If get_X_y uses a stride method, check that X and y contains the optimal number of sample each
    """
    X, y = get_X_y(data_monotonic_increase, **TRAIN)

    # Complex formula below retro-engineered from `create_dummy_tests.ipynb`
    expected_len = math.ceil(
        (len(data_monotonic_increase) \
            - (TRAIN['input_length']  -1) \
            - (TRAIN['output_length'] -1) \
            - TRAIN['horizon']
        ) / TRAIN["stride"]
    )
    assert len(X) == expected_len, "you may have not generated the optimal number of samples, given the stride chosen"
    assert len(y) == expected_len, "you may have not generated the optimal number of samples, given the stride chosen"

def test_no_data_leak(data_monotonic_increase):
    """Test that the time gap between the last timestep of `y_train` and the first timestep of `y_test`
    is at least as big as the forecast horizon
    according to 'https://raw.githubusercontent.com/lewagon/data-images/master/DL/rnn-3.png'
    """

    data_train, data_test = train_test_split(data_monotonic_increase, **TRAIN)
    X_train, y_train = get_X_y(data_train, shuffle=False, **TRAIN)
    X_test, y_test = get_X_y(data_test, shuffle=False, **TRAIN)

    y_train_last_seen_timestep = np.max(y_train)  # OR y_train[-1].flat[-1]
    y_test_first_seen_timestep = np.min(y_test)  # OR y_test[0].flat[0]
    gap = y_test_first_seen_timestep - y_train_last_seen_timestep
    # Note: for strides = 1, the inequality below must be an exact equality, but we don't need to test that to ensure no data leak.
    assert gap >= TRAIN["horizon"], "❗️❗️ Data leak detected between (X_train, y_train) and (X_test, y_test)❗️❗️ "
