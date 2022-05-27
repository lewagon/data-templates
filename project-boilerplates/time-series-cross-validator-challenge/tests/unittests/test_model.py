import pytest
from ts_boilerplate.model import fit_model, get_model, predict_output

def test_model_has_correct_output_shape(X_y_zeros_and_ones):
    X, y = X_y_zeros_and_ones
    model = get_model(X,y)
    y_pred = predict_output(model, X)
    assert y_pred.shape == y.shape

@pytest.mark.slow
def test_model_can_fit(X_y_zeros_and_ones):
    """Check that the model can fit without crashing"""
    X, y = X_y_zeros_and_ones
    model = get_model(X,y)
    fit_model(model, X, y, verbose=0)
