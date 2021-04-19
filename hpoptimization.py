import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from kerastuner.tuners import RandomSearch
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import deque

N_STEPS = 50
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
PERIOD = '1y'
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
INTERVAL = '1d'
# Lookup step, 1 is the next day
LOOKUP_STEP = 10
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.3
# features to use
FEATURE_COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
# date now
date_now = time.strftime("%Y-%m-%d")
# ticker
ticker = 'TSLA'
# cell
cell = LSTM
# loss
loss = 'huber_loss'
# specify whether to use bidirectional neurons
bidirectional = True

def build_model1(hp1):
    model = Sequential()
    for i in range(hp1.Int('units', min_value=1, max_value=10)):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(return_sequences=True, units=hp1.Int('units', min_value=32,
                                                                                 max_value=512,
                                                                                 step=32,
                                                                                 ))))
            else:
                model.add(cell(return_sequences=True, units=hp1.Int('units', min_value=32,
                                                                   max_value=512,
                                                                   step=32, )))
        elif i == i - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(return_sequences=True,
                                             units=hp1.Int('units', min_value=32,
                                                          max_value=512,
                                                          step=32, ))))
            else:
                model.add(cell(return_sequences=True, units=hp1.Int('units', min_value=32,
                                                                   max_value=512,
                                                                   step=32, )))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(return_sequences=True,
                                             units=hp1.Int('units', min_value=32,
                                                          max_value=512,
                                                          step=32, ))))
            else:
                model.add(cell(return_sequences=True,
                               units=hp1.Int('units', min_value=32,
                                            max_value=512,
                                            step=32, )))
        # add dropout after each layer
        tf.keras.layers.Dropout(
            hp1.Float('dropout', min_value=0, max_value=0.5, step=0.1, default=0.5))(i)
        model.add(Dense(units=hp1.Int('units', min_value=3, max_value=10))),
        model.add(Dense(1)),

    model.compile(loss="huber_loss", metrics=["accuracy"], optimizer=tf.keras.optimizers.Adam((
        hp1.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))))
    return model


tuner = RandomSearch(
    build_model1,
    objective="accuracy",
    seed=42,
    max_trials=5,
    executions_per_trial=2,
    tune_new_entries=True,
    directory= 'Users/kylehammerberg/Desktop/tunermain',
    project_name= 'tuner1',
    allow_new_entries=True)

import yfinance as yf
import numpy as np


def load_data(ticker, period, interval, n_steps=100, scale=True, shuffle=True, lookup_step=10, test_size=.2,
              feature_columns=['Close', 'Volume', 'Open', 'High', 'Low']):
    df = yf.download(tickers=ticker, period=period, interval=interval,
                     group_by='ticker',
                     # adjust all OHLC automatically
                     auto_adjust=True, prepost=True, threads=True, proxy=None)

    result = {}
    result['df'] = df.copy()

    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                shuffle=shuffle)
    # return the result
    return result


data = load_data(ticker, PERIOD, INTERVAL, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                 feature_columns=FEATURE_COLUMNS, shuffle=False)

train_ds, test_ds = data['X_test'], data['y_test']
train_ds = train_ds.astype('float32')
test_ds = train_ds.astype('float32')

print(f'test_ds: {test_ds}')

print(f'train_ds: {train_ds}')


N_EPOCH_SEARCH = 100

tuner.search(train_ds, test_ds, epochs=N_EPOCH_SEARCH, validation_split=0.1)

tuner.search_space_summary()

# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=2)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(train_ds, test_ds)
