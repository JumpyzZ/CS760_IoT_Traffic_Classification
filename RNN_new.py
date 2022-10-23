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
    def __init__(self, X_train):
        self.model = Sequential()
        #  1st RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        # 2nd RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        # 3rd RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        # 4th RNN layer and Dropout regularization
        self.model.add(SimpleRNN(units=50))
        self.model.add(Dropout(0.2))
        # output layer
        self.model.add(Dense(units=1))

    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X_train, y_train):
        X_train = np.array(X_train).reshape([-1, X_train.shape[1], 1])
        self.model.fit(X_train, y_train)

    def predict(self, X_eval):
        X_eval = np.array(X_eval).reshape([-1, X_eval.shape[1], 1])
        return self.model.predict_classes(X_eval)


""""class RNNClass(tf.keras.Model):
    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self.layer1 = SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(n, 1))
        self.layer2 = Dropout(0.2)
        self.layer3 = SimpleRNN(units=50, activation='relu', return_sequences=True)
        self.layer4 = Dropout(0.2)
        self.layer5 = SimpleRNN(units=50, activation='relu', return_sequences=True)
        self.layer6 = Dropout(0.2)
        self.layer7 = SimpleRNN(units=50)
        self.layer8 = Dense(units=1)

    def call(self, inputs, training=None, mask=None):
        return self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layers2(self.layers1(self.reshape(inputs)))))))))

    def fit(self, X_train, y_train):
        super.fit(X_train, y_train)

    def reshape(self, X):
        return X.reshape([-1, X.shape[1], 1])"""""