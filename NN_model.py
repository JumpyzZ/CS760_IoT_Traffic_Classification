import tensorflow as tf
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from preprocess import *
from typing import Union


Numeric = Union[float, int]
Array = Union[np.array, list]


@dataclass
class MetricHistory:
    metrics: dict
    epoch_lim: float = field(default=0)

    def update(self, new_metrics: Array) -> None:
        assert len(new_metrics) == len(self.metrics)

        for n, key in enumerate(self.metrics):
            self.metrics[key].append(new_metrics[n])

        self.epoch_lim += 1


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = {'binary_cross_entropy': tf.keras.metrics.BinaryCrossentropy()}
        self.history = MetricHistory({'binary_cross_entropy': []})

    def call(self, y_true: Numeric, y_pred: Numeric) -> list:
        array = []

        for key in self.metrics:
            m = self.metrics[key](y_true, y_pred)
            array.append(m)

        self.history.update(array)

        return array


def threshold(predictions: np.array) -> np.array:
    return np.array([1 if t > 0.5 else 0 for t in predictions])


class CustomNN(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer='l1')
        self.dense2 = tf.keras.layers.Dense(6, activation=tf.nn.relu, kernel_regularizer='l2')
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer='l2')
    
    def call(self, inputs: np.array) -> np.array:
        features1 = self.dense1(inputs)
        features2 = self.dense2(features1)
        features3 = self.dense3(features2)

        return features3
