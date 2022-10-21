import pandas as pd
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from numpy import loadtxt
from keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn import datasets
from keras.layers import *
from keras import Model
from keras.utils import plot_model,to_categorical


def CNN_Model(fea_cnt, numb):
    K.clear_session() # clear memory
    inputs = Input(shape=(fea_cnt,), dtype='float32')
    embds = Dense(64, activation='selu')(inputs)
    embds = Reshape((8, 8, 1))(embds)
    embds = Conv2D(1, (4, 4), strides=(1, 1), padding="valid")(embds)
    embds = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(embds)
    embds = Flatten()(embds)
    embds = Dense(64, activation='selu')(embds)
    embds = Dense(32, activation='selu')(embds)
    outputs = Dense(numb-1, activation='sigmoid')(embds)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def test():
    dataX = loadtxt('Dataset/UNSW-NB15 - CSV Files/UNSW-NB15_1.csv', skiprows=1, delimiter=',', usecols=range(5, 43))
    dataY = loadtxt('Dataset/UNSW-NB15 - CSV Files/UNSW-NB15_1.csv', skiprows=1, delimiter=',', usecols=44)

    fea = len(dataX[0]) # number of features
    numc = 2 # number of class label 0 for normal 1 for attack
    dataY = dataY.astype(int) # convert float to int for class label

    Y = np.array([to_categorical(y, numc, dtype='int32') for y in dataY])

    model = CNN_Model(fea, numc)
    X, Y = shuffle(dataX, Y, random_state=1234)
    validation_split = 0.2

    train_X = X[:int(len(X)*(1-validation_split))]
    train_Y = Y[:int(len(Y)*(1-validation_split))]
    test_X = X[int(len(X)*(1-validation_split)):]
    test_Y = Y[int(len(Y)*(1-validation_split)):]

    epochs = 12
    batch_size = 4
    model.fit(train_X, train_Y
              ,batch_size = batch_size
              ,epochs = epochs
              ,validation_data= [test_X,test_Y])

    plot_model(model, to_file='./model.png', show_shapes=True, dpi=300)
    score = model.evaluate(test_X, test_Y)

