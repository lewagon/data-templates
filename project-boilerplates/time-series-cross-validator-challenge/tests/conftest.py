import pytest
import numpy as np
from ts_boilerplate.generate_dummy_data import generate_data_monotonic_increase, generate_data_zeros_and_ones, generate_X_y_zeros_and_ones
from typing import Tuple

@pytest.fixture(scope="session")
def data_monotonic_increase() -> np.ndarray:
    return generate_data_monotonic_increase()

@pytest.fixture(scope="session")
def data_zeros_and_ones() -> np.ndarray:
    return generate_data_zeros_and_ones()


@pytest.fixture(scope="session")
def X_y_zeros_and_ones() -> Tuple[np.ndarray]:
    return generate_X_y_zeros_and_ones()
