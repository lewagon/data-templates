
# TODO
## V1: MVP
- [x] Basic training & cross-val routes, with test
- [x] Make model fit well for univariate, multivariate (n_tagets >1) & sequences (output_sequence_lenght>1)
- [x] Make tests pass for stride > 1
- [x] Add tests about the model (shape of prediction, etc)

## V2 : must have before student release
- [x] Add `main.backtesting` as main route (see [darts.model.historical_forecasts()](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel.historical_forecasts): A a very important concept to teach to students)
- [ ] Refacto `model.py`
  - [ ] Rename `pipeline.py` because it may comprises the pre-processing such as scaling etc...
  - [ ] Turn into a class `TsPipeline()` instead of pure functions

## V3 : nice to have
- [ ] Integrate package as part of the ML Ops lifecycle
  - [ ] track & save experiment results
  - [ ] ...

- [ ] Add tests for future-covariates
- [ ] Create Makefile
  - [ ] Include DAG of the project
- [ ] publish to lewagon community
