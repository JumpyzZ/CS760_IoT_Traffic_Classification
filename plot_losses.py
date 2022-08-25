from train_model import *
# from NN_model import *


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


