import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from preprocess import *


EPOCHS = 32
BATCH_SIZE = 256
TIME_STEPS = 16
UNITS = 32


def LSTM_model(TIME_STEPS, UNITS):
    model = Sequential()
    model.add(LSTM(units = UNITS, input_shape = (TIME_STEPS, X_train.shape[1])))
    model.add(Dense(1, activation = 'softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = preprocess_UNSW()
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


X_train_ = np.lib.stride_tricks.sliding_window_view(X_train, (TIME_STEPS,X_train.shape[1]))
X_train_ = X_train_.reshape(X_train_.shape[0], X_train_.shape[2], X_train_.shape[3])
X_train_ = X_train_[0:-1,:]
y_train_ = np.array(y_train)[TIME_STEPS:,:]
print(X_train.shape, X_train_.shape)
print(y_train.shape, y_train_.shape)


X_test_ = np.lib.stride_tricks.sliding_window_view(X_test, (TIME_STEPS,X_test.shape[1]))
X_test_ = X_test_.reshape(X_test_.shape[0], X_test_.shape[2], X_test_.shape[3])
X_test_ = X_test_[0:-1,:]
y_test_ = np.array(y_test)[TIME_STEPS:,:]
print(X_test.shape, X_test_.shape)
print(y_test.shape, y_test_.shape)


lstm_model = LSTM_model(TIME_STEPS = 16, UNITS = 32)
lstm_model.fit(X_train_, y_train_, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1)


loss, acc = lstm_model.evaluate(X_test_, y_test_, batch_size = BATCH_SIZE, verbose=2)
