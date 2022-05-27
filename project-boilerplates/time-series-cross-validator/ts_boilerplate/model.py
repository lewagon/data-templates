import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Reshape, Lambda, Input
from tensorflow.keras import Model
from ts_boilerplate.params import DATA, TRAIN

# TODO: Should we add here the preprocessing? into a class called "pipeline"?
# TODO: Should we refacto in a class ? Probably!


def get_model(X_train, y_train):
    """Instanciate, compile and and return the model of your choice"""
    # $CHALLENGIFY_BEGIN

    # BASELINE: PREDICT LAST VALUE - ZERO TRAINABLE WEIGHTS
    input = Input(shape=X_train.shape[1:])
    # Take last temporal values of the targets, and duplicate it as many times as `output_length`
    x = Lambda(
        lambda x: tf.repeat(
            tf.expand_dims(tf.gather(x[:, -1, :], indices=DATA['target_column_idx'], axis=1), axis=1),
            repeats=TRAIN['output_length'],
            axis=1)
        )(input)
    output = Reshape(y_train.shape[1:])(x)
    model = Model(input, output)

    # # THE SIMPLEST OF ALL POSSIBLE RNN
    # model = tf.keras.Sequential()
    # model.add(SimpleRNN(1, activation='tanh', input_shape=X_train.shape[1:]))
    # model.add(Dense(TRAIN['output_length'] * DATA["n_targets"], activation='linear'))
    # model.add(Reshape(y_train.shape[1:]))
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=tf.keras.metrics.MAPE)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=tf.keras.metrics.MAPE)
    return model
    # $CHALLENGIFY_END


def fit_model(model, X_train, y_train, **kwargs):
    """Fit the `model` object, including preprocessing if needs be"""
    # $CHALLENGIFY_BEGIN
    verbose = kwargs.get("verbose", 0)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=2,
                                          verbose=verbose,
                                          mode='min',
                                          restore_best_weights=True)
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        batch_size=16,
                        validation_split=0.3,
                        callbacks=[es],
                        verbose=verbose)
    return history
    # $CHALLENGIFY_END


def predict_output(model, X_test):
    """Return y_test. Include preprocessing if needs be"""
    # $CHALLENGIFY_BEGIN
    y_pred = model.predict(X_test)
    return y_pred
    # $CHALLENGIFY_END
