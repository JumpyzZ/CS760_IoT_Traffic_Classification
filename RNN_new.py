import pandas as pd
from tensorflow.python.keras.layers import Dropout,SimpleRNN
import tensorflow as tf
import numpy as np


def RNN_base(X_train: pd.DataFrame):
    model = tf.keras.models.Sequential()
    #  1st RNN layer and Dropout regularization
    # model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(x_train_.shape[1], 1)))
    X = np.array(X_train).reshape([-1, X_train.shape[1], 1])
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=X))
    model.add(Dropout(0.2))
    # 2nd RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    # 3rd RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    # 4th RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50))
    model.add(Dropout(0.2))
    # output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

