import tensorflow as tf
import numpy as np 

class CustomLoss(tf.Loss):
    def __init__(self) -> None:
        super().__init__()
        self.accuracies = []
    
    def call(self, y_true, y_pred):
        residual = np.abs(y_true - y_pred)
        self.accuracies.append(residual)


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


custom_nn = CustomNN()
custom_nn.compile(optimiser='adam', loss=CustomLoss)


