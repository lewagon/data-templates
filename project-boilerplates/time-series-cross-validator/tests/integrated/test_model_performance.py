import pytest
from ts_boilerplate.main import train


@pytest.mark.optional
@pytest.mark.slow
def test_model_can_fit_well_enough(data_zeros_and_ones):
    """Check that the model can fit, with MAPE lower than some threshold on dummy dataset of zeros and ones,"""
    metrics = train(data_zeros_and_ones)
    print("#### metrics ", metrics)
    assert metrics < 5, "your model does not seem to be able to fit well enough even a very easy dataset"
