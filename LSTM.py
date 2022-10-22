import numpy as np
import tensorflow as tf
from preprocess import *


def LSTM_model(time_steps: int, units: int, n: int):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=units, input_shape=(time_steps, n)))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    return model


def test():
    epochs = 32
    BATCH_SIZE = 256
    time_steps = 16
    cells = 32

    X_train, X_test, y_train, y_test = preprocess_UNSW()

    #X_train = X_train.iloc[0:50, :]  # for speed
    #y_train = y_train.iloc[0:50, :]
    #X_eval = X_test.iloc[0:50, :]
    #y_eval = y_test.iloc[0:50, :]

    X_train_ = np.lib.stride_tricks.sliding_window_view(X_train, (time_steps, X_train.shape[1]))
    X_train_ = X_train_.reshape(X_train_.shape[0], X_train_.shape[2], X_train_.shape[3])
    X_train_ = X_train_[0:-1,:]
    y_train_ = np.array(y_train)[time_steps:,:]

    X_test_ = np.lib.stride_tricks.sliding_window_view(X_test, (time_steps, X_test.shape[1]))
    X_test_ = X_test_.reshape(X_test_.shape[0], X_test_.shape[2], X_test_.shape[3])
    X_test_ = X_test_[0:-1,:]
    y_test_ = np.array(y_test)[time_steps:,:]

    n = X_train.shape[0]
    m = X_train.shape[1]

    lstm_model = LSTM_model(16, 32, X_train.shape[1])
    lstm_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
    lstm_model.fit(X_train.values.reshape((m, time_steps, n)), y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=1)

    loss, acc = lstm_model.evaluate(X_test_, y_test_, batch_size=BATCH_SIZE, verbose=2)

class lstmClass:
    def __init__(self):
        self.__epochs = 32
        self.__BATCH_SIZE = 256
        self.__time_steps = 16
        self.__cells = 32
        X_train, X_test, y_train, y_test = preprocess_UNSW()

        #X_train = X_train.iloc[0:50, :]  # for speed
        #y_train = y_train.iloc[0:50, :]
        #X_eval = X_test.iloc[0:50, :]
        #y_eval = y_test.iloc[0:50, :]

        X_train_ = np.lib.stride_tricks.sliding_window_view(X_train, (self.__time_steps, X_train.shape[1]))
        X_train_ = X_train_.reshape(X_train_.shape[0], X_train_.shape[2], X_train_.shape[3])
        X_train_ = X_train_[0:-1,:]
        y_train_ = np.array(y_train)[self.__time_steps:,:]

        X_test_ = np.lib.stride_tricks.sliding_window_view(X_test, (self.__time_steps, X_test.shape[1]))
        X_test_ = X_test_.reshape(X_test_.shape[0], X_test_.shape[2], X_test_.shape[3])
        X_test_ = X_test_[0:-1,:]
        y_test_ = np.array(y_test)[self.__time_steps:,:]

        self.__n = X_train.shape[0]
        self.__m = X_train.shape[1]
        self.__lstm_model = LSTM_model(16, 32, X_train.shape[1])
        self.__lstm_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])

    def fit(self, x, y, verbose):
        self.__lstm_model.fit(x.values.reshape((self.__m, self.__time_steps, self.__n)), y, epochs=self.__epochs, batch_size=self.__BATCH_SIZE, verbose=verbose)
    
    def compile(self):
        self.__lstm_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])