"""Tests that main route run without raising exceptions"""

import pytest
from ts_boilerplate.main import backtest, train, cross_validate

@pytest.mark.slow
def test_main_route_train(data_monotonic_increase):
    train(data_monotonic_increase)

@pytest.mark.slow
def test_main_route_cross_validate(data_monotonic_increase):
    cross_validate(data_monotonic_increase)

@pytest.mark.slow
def test_backtest(data_monotonic_increase):
    backtest(data_monotonic_increase, print_metrics=False, plot_metrics=False)
