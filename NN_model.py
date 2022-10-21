import tensorflow as tf
from preprocess import *
from typing import Union


Numeric = Union[float, int]
Array = Union[np.array, list]


class CustomNN(tf.keras.Model):
    def __init__(self, n: int = 10, weight_init=None) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(n + 10, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(n + 10, activation=tf.nn.selu)
        self.dense3 = tf.keras.layers.Dense(5, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    
    def call(self, inputs: np.array) -> np.array:
        features1 = self.dense1(inputs)
        features2 = self.dense2(features1)
        features3 = self.dense3(features2)
        features4 = self.dense4(features3)

        return features4


def myround(predictions: np.array) -> np.array:
    return np.array([1 if t > 0.5 else 0 for t in predictions])





