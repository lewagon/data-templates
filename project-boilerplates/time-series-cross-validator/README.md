This is a **boilerplate** repo for a machine-learning project involving **Time Series forecasting**.

In particular

- It provides a **cross-validation framework** to ensure model are tested thoroughly and without data leakage
- It is agnostic of the type of model involved
- It is well suited for short research projects, typical of few-weeks coding bootcamps such as Le Wagon DataScience

# Detailed package workflow

## Architecture
- `ts_boilerplate` package
  - `main.py` comprises the main routes to be called from the CLI (`train`, `cross-validate`, `backtest`)
  - `params.py` contains project-level global variable to be set manually
<br>

- `data` folder contains
  - `raw` and `clean` folder should contain **2D arrays `data` time-series**, with (axis 0) representing timesteps integer, and (axis 1) columns containing tagets and covariates, as per [picture](https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true)
    ```python
    data.shape = (length, n_targets+n_covariates)
    ```
  - `Xy` may persist your tuple (X,y) of **3D arrays** training sets to be fed to your models if you want to store them to avoid preprocessing multiple times.
    ```python
    X.shape = (n_samples, input_length, n_covariates)
    y.shape = (n_samples, output_length, n_targets)
    ```
- `notebooks`
  - `test_package.ipynb` will help you understand how the package and the tests have been built.
  - `tutorial_ts_forecasting.ipynb` is a recommended read before diving into this project. It contains visuals that will help you fill global project params and understand naming conventions

<br>

- `tests` folder detailed below

## How to test your code?
First of all, fill `ts_boilerplate/params.py` corresponding to your true project speficities

Then, run this in your terminal from the root project folder to check your code
- `pytest`
- `pytest -m "not optional"`  to only check mandatory tests
- `pytest -m "not optional" -m "not slow"` to also avoid tests that may be slow (involving fitting your model)

