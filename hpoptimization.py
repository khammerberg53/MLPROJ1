import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
from collections import deque
import kerastuner as kt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import yfinance as yf

#units = neurons
def load_data(ticker, period, interval, n_steps=200, scale=True, shuffle=True, lookup_step=30, test_size=.2,
              feature_columns=['Close', 'Volume', 'Open', 'High', 'Low']):
    '''
    :param ticker: Ticker you want to load, dtype: str
    :param period: Time period you want data from, dtype: str(options in program)
    :param interval: Interval for data, dtype:str
    :param n_steps: Past sequence length used to predict, default = 50, dtype: int
    :param scale: Whether to scale data b/w 0 and 1, default = True, dtype: Bool
    :param shuffle: Whether to shuffle data, default = True, dtyper: Bool
    :param lookup_step: Future lookup step to predict, default = 1(next day), dtype:int
    :param test_size: ratio for test data, default is .2 (20% test data), dtype: float
    :param feature_columns: list of features fed into the model, default is OHLCV, dtype: list
    :return:
    '''
    df = yf.download(tickers=ticker, period=period, interval=interval,
                     group_by='ticker',
                     # adjust all OHLC automatically
                     auto_adjust=True, prepost=True, threads=True, proxy=None)

    result = {}
    result['df'] = df.copy()
    ### preview data frame before preprocessing
    print(df)

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
                                                                               test_size=test_size, shuffle=shuffle)
    # return the result
    return result

print(f'{result}')
