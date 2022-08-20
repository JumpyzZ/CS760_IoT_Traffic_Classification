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
    epoch_lim: int = field(default=0)

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


def train_nn(network: CustomNN, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    epoch_length = 5 # np.floor(len(X_train) / 100) + 1
    epoch_batch = [[], []]  # collection of (x, y) data points

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss')   # callback gives extra stopping criteria

    # train network on batch and evaluate performance
    for n, xtr, ytr in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy()):
        epoch_batch[0].append(xtr)
        epoch_batch[1].append(ytr)
        if n % epoch_length == 0:
            network.fit(pd.DataFrame(epoch_batch[0]), pd.DataFrame(epoch_batch[1]), callbacks=[early_stopping])


def plot(history: MetricHistory):
    plt.grid()
    plt.title('Neural Network Evaluation', fontsize=25)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Binary Cross Entropy', fontsize=20)

    plt.show()


def controller():
    custom_nn = CustomNN()
    custom_loss = CustomLoss()
    custom_nn.compile(optimizer="Adam", loss=custom_loss)

    # load data if it exists
    data_path = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files',
                             'a part of training and testing set', 'UNSW_NB15_PREPROCESSED.csv'])

    UNSW = pd.read_csv(data_path)
    y = UNSW.iloc[:, -1:]
    X = UNSW.iloc[:, :-1]

    train_nn(custom_nn, X, y)
    history = custom_loss.history
    print(history)
    plot(history)
    # print(history)

    # plot_history(train_nn(custom_nn))


controller()

