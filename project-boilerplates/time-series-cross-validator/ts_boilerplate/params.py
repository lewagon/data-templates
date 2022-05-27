import os
from dotenv import load_dotenv

## CREDENTIALS AND PATHS
load_dotenv()
API_KEY = os.getenv('API_KEY')

## DIR PARAMS
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'data.csv')

# 👇 Please fill these global variable below very carefully, in order to create tests related to your problem👇
# cf: https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true
DATA = dict(
    length = 500, # How many timesteps does your dataset contains?
    n_covariates = 3, # number of past covariates, excluding target time series. Our tests do not support future_covariate yet.
    target_column_idx = [0,1] # List of index(es) of target column(s) in your dataset. e.g [0] for Mono-target problem, e.g. [0,1,4] for multi-variate targets problem. Note that past targets values will also be used as features X.
)
DATA['n_targets'] = len(DATA['target_column_idx']) # number of target time series to predict.

TRAIN = dict(
    horizon = 4, # start prediction xxx timestep ahead
    input_length = 10, # Length (in time) of each sequences that will be seen by the model (X.shape[1])
    output_length = 7, # Length (in time) of prediction (y.shape[1])
    stride = 1, # Integer used to create all pairs of sample (Xi, yi) by sliding in each data fold. Use `None` if you don't plan to use any sliding method in data.get_X_y
    train_test_ratio = 0.7, # ratio of train / (train+test) length in each fold
)

CROSS_VAL = dict(
    fold_length = 200,
    fold_stride = 100,
)
