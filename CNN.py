import pdb
import os
import re
import pandas as pd
import numpy as np
import time

from numpy import mean
from numpy import dstack
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn import model_selection
from numpy import loadtxt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_path_of_csv_files(top) -> list:
    """
    get all csv file path in 'top' and all its subfolers
    :param top: root dir string
    :return: ['Dataset\\N_BaIot\\Danmini_Doorbell\\benign_traffic.csv', ...]
    """

    path_of_csv_files = []
    for path_dir_files in os.walk(top):
        for file_name in path_dir_files[2]:
            if re.search('^(?!\.)(.*?).csv',
                         file_name) != None and file_name != 'demonstrate_structure.csv' and file_name != 'N_BaIot_PREPROCESSED_FULL.csv':
                path_of_csv_files.append(path_dir_files[0] + os.sep + file_name)

    return path_of_csv_files

def get_file_type(file_path: str) -> dict:
    """
    get device name and traffic type based on file path
    :param file_path: a single csv path
    :return: {'file_path': 'Dataset\\N_BaIot\\SimpleHome_XCS7_1003_WHT_Security_Camera\\mirai_attacks\\udpplain.csv',
              'device_name': 'SimpleHome_XCS7_1003_WHT_Security_Camera',
              'traffic_type': 'mirai_attacks-udpplain'}
    """

    device_name = file_path.split(os.sep)[2]
    traffic_type = ''

    if file_path.count(os.sep) == 3 and file_path.split(os.sep)[-1] == 'benign_traffic.csv':
        traffic_type = 'benign_traffic'

    if file_path.count(os.sep) == 4:
        traffic_type = file_path.split(os.sep)[3] + '-' + file_path.split(os.sep)[4].replace('.csv', '')

    return {'file_path': file_path,
            'device_name': device_name,
            'traffic_type': traffic_type}

def get_N_BaIot() -> pd.DataFrame:
    """
    :return: a dataframe
    """

    count = 1
    dataset_dir = 'Dataset' + os.sep + 'N_BaIot'
    file_list = [get_file_type(file_path) for file_path in get_path_of_csv_files(dataset_dir)]
    data = pd.DataFrame()
    for file in file_list:
        print('(loading {c}/{f})'.format(c=count, f=len(file_list)), file['file_path'])
        df_tmp = pd.read_csv(file['file_path'])
        df_tmp['device_name'] = file['device_name']
        df_tmp['traffic_type'] = file['traffic_type']
        if len(data.columns) == 0:
            data = df_tmp
        else:
            data = pd.concat([data, df_tmp])
        count += 1

    return pd.DataFrame(data)

# generate new dataset for CNN includ benign traffic and specific attack type from temp3 excluding benigh traffic
def X_Y_to_CSV(attack_name, device_name):
    temp2_new = temp2.drop(columns=['device_name'])
    df = temp2_new[(temp2_new['traffic_type'] == 'benign_traffic') | (temp2_new['traffic_type'] == attack_name)]
    label = df['traffic_type']
    df = df.drop(columns=['traffic_type'])

    # write X into csv
    df.to_csv("Dataset\\N_BaIot\\" + device_name + "\\CNN_Dataset\\X_" + attack_name + ".csv", index=None, header = None)
    label = pd.DataFrame(label)
    # write label into csv
    for index, row in label.iterrows():
        if row['traffic_type'] == 'benign_traffic':
            row['traffic_type'] = 0
        else:
            row['traffic_type'] = 1
    label.to_csv("Dataset\\N_BaIot\\" + device_name + "\\CNN_Dataset\\Y_" + attack_name + ".csv", index=None, header = None)

def Load(attack_name, split=0.2, seed=760):

    print("Loading data ...")
    dataX = loadtxt('Dataset/N_BaIot/Danmii_Doorbell/CNN_Dataset/X_' + attack_name + '.csv', delimiter=',')
    dataY = loadtxt('Dataset/N_BaIot/Danmii_Doorbell/CNN_Dataset/Y_' + attack_name + '.csv', delimiter=',')

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


def CNN_Model(trainX, trainY, testX, testY):
    n_features, n_added_dimension, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = Sequential()
    # First Conv Layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_features, n_added_dimension)))
    # Second Conv Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # Maxpooling Layer
    model.add(MaxPooling1D(pool_size=2))
    # Third Conv Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    # Maxpooling Layer
    model.add(MaxPooling1D(pool_size=2))
    # flatten layer
    model.add(Flatten())
    # first dense layer
    model.add(Dense(50, activation='relu'))
    # second dense layer
    model.add(Dense(n_outputs, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)

    return model