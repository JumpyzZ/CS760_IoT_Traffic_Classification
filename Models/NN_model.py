import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn as sk 

from dataclasses import dataclass, field
from typing import Union


@dataclass
class MetricHistory:
    epochs: np.array 
    metric: np.array

    def update(self, array: Union[list, np.array]) -> None:
        for n, m in enumerate(array):
            self.metric.append(m)
            self.epochs.append(n)
    


class CustomLoss(tf.Loss):
    def __init__(self) -> None:
        super().__init__()
        self.metrics = {'accuracey': tf.keras.metrics.Accuracy()} # idk 
        self.history = MetricHistory()

    
    def call(self, y_true, y_pred) -> None:
        array = []
        
        for key in self.metrics:
            m = self.metrics[key](y_pred, y_true)
            array.append(m)
        
        self.history.update(array)

        return array


class CustomNN(tf.Model):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer='l1')
        self.dense2 = tf.keras.layers.Dense(6, activation=tf.nn.relu, kernel_regularizer='l2')
        self.dense3 = tf.keras.layers.Dense(4, activation=tf.nn.softmax, kernel_regularizer='l2')

        self.statistics = [tf.keras.metrics.Accuracy(name='accuracy')]
    
    def call(self, x, inputs) -> np.array:
        self.features1 = self.dense1(inputs)
        self.features2 = self.dense2(self.features1)
        self.features3 = self.dense3(self.features2)

        return self.features3


def train_nn(network: CustomNN, X_train: pd.DataFrame, y_train: pd.DataFrame) -> MetricHistory:
    epoch_length = np.floor(len(X_train) / 100)
    epoch_batch = [[], []]

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', paitence=2)

    # train on epoch data, reset test data. Check the documentation for instructions on custom training please. 
    # pretty sure there are serveral mistakes in this, it was late -- Thomas 
    for n, xtr, ytr in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy()):
        epoch_batch[0].append(xtr)
        epoch_batch[1].append(ytr)
        if n % epoch_length == 0:
             history = network.fit(pd.DataFrame(epoch_batch[0]), pd.DataFrame(epoch_batch[1]), callbacks=[early_stopping])     

    return network.history

def plot_history(history: MetricHistory): # keep it simple for now 
    plt.plot(history.epochs, history.metric[0])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracey')
    plt.show()



custom_nn = CustomNN()
custom_nn.compile(optimiser='adam', loss=CustomLoss())

# read data 

plot_history(train_nn(custom_nn))




