import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,SimpleRNN
from preprocess import *


def RNN_base(hidden_units:int, dense_units:int, X_train:pd.DataFrame):
        model = Sequential()
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
        model.add(Dense(units=1), activation='sigmoid')

        return model





