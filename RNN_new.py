import pandas as pd
from tensorflow.python.keras.layers import Dropout,SimpleRNN,Dense
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
import numpy as np
from preprocess import preprocess_UNSW


def RNN_base(X_train: pd.DataFrame):
    model = tf.keras.models.Sequential()
    #  1st RNN layer and Dropout regularization
    # self.model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(x_train_.shape[1], 1)))
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

class RNNClass:
    def __init__(self):
        self.model
    
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss = loss, optimizer = optimizer, metrics=metrics)

    def fit(self, X_train, y_train):
        sample = X_train.shape[0]
        features = X_train.shape[1]
        self.model = Sequential()
        #  1st RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True, input_shape= (X_train.shape[1],1)))
        self.model.add(Dropout(0.2))
        # 2nd RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        # 3rd RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        # 4th RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units = 50))
        self.model.add(Dropout(0.2))
        # output layer
        self.model.add(Dense(units = 1))
        self.model.fit(X_train, y_train, epochs=32, batch_size=128,verbose = 1)

    def predict(self, X_eval):
        sample = self.X_eval.shape[0]
        features = self.X_eval.shape[1]
        X_eval = np.array(X_eval).reshape([-1, X_eval.shape[1], 1])
        return self.model.predict_classes(X_eval)