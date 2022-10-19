import tensorflow as tf

from numpy import dstack
from numpy import loadtxt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def Load(split=0.2, seed=760):

    print("Loading data ...")
    dataX = loadtxt('Dataset/UNSW-NB15 - CSV Files/UNSW_NB15.csv', delimiter=',', usecols=range(43))
    dataY = loadtxt('Dataset/UNSW-NB15 - CSV FilesUNSW_NB15.csv', delimiter=',', usecols=44)

    # split data into training and test set
    print("Spliting data ...")
    trainX, testX = train_test_split(dataX, test_size=split, random_state=seed, shuffle=True)
    trainY, testY = train_test_split(dataY, test_size=split, random_state=seed, shuffle=True)

    # dstack for training and test set
    list_train = list()
    list_test = list()
    list_train.append(trainX)
    list_test.append(testX)

    print("Stacking data ...")
    trainX = dstack(list_train)
    testX = dstack(list_test)

    print("Category data ...")
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    print('trainX.shape: ', trainX.shape)
    print('testX.shape: ', testX.shape)
    print('trainy.shape: ', trainY.shape)
    print('testy.shape: ', testY.shape, '\n')

    return trainX, trainY, testX, testY


def CNN_Model(trainX, trainY):
    n_features, n_added_dimension, n_outputs = trainX.shape[1], trainX.shape[0], trainY.shape[1]
    model = tf.keras.models.Sequential()
    # First Conv Layer
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_features, n_added_dimension)))
    # Second Conv Layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    # Maxpooling Layer
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    # Third Conv Layer
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    # Maxpooling Layer
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    # flatten layer
    model.add(tf.keras.layers.Flatten())
    # first dense layer
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    # second dense layer
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))    # compile model

    return model
