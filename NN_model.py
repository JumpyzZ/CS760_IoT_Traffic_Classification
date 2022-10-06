import tensorflow as tf
from preprocess import *
from typing import Union


Numeric = Union[float, int]
Array = Union[np.array, list]


class CustomNN(tf.keras.Model):
    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(n, activation=tf.nn.relu, kernel_regularizer='l1')
        self.dense2 = tf.keras.layers.Dense(6, activation=tf.nn.relu, kernel_regularizer='l2')
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer='l2')
    
    def call(self, inputs: np.array) -> np.array:
        features1 = self.dense1(inputs)
        features2 = self.dense2(features1)
        features3 = self.dense3(features2)

        return features3


def myround(predictions: np.array) -> np.array:
    return np.array([1 if t > 0.5 else 0 for t in predictions])





