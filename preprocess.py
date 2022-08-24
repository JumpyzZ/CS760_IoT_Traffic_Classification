import pdb
import os
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
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


def preprocess(data: pd.DataFrame, num_features: int = 10) -> tuple([pd.DataFrame, pd.DataFrame]):
    """
    Perform preprocessing transformations on given data
    :param df: Dataframe to be processed
    :return: Dataframe of transformed data 
    """

    data = pd.DataFrame(shuffle(data.values), columns=data.columns)   # stop label of the same class clustering
    y = data['label']
    data = data.drop(columns='label')
    array1 = PCA(n_components=min(data.shape[1], num_features)).fit_transform(data)  # perform PCA on standardized data
    array2 = StandardScaler().fit_transform(array1)  # standardize data
    array3 = array2 + np.random.normal(0, 0.3, size=array2.shape)  # add gaussian noise

    return pd.DataFrame(array3), y 


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


def get_UNSW() -> pd.DataFrame:
    """
    :return: The UNSW dataset
    """

    train_path = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_training-set.csv'])
    test_path = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_testing-set.csv'])

    # fetch data
    df = pd.concat([pd.read_csv(train_path, header=0), pd.read_csv(test_path, header=0)], axis=0, ignore_index=False)
    df = df.drop(columns=['id'])

    # get the names of non-numerical data columns
    string_columns = []
    for n, val in enumerate(df.values[0]):
        if not (isinstance(val, int) or isinstance(val, float)):
            string_columns.append(df.columns[n])

    # turn non-numeric columns entries into integer entries
    df[string_columns] = OrdinalEncoder().fit_transform(df[string_columns])

    # save column names
    path_mapping_csv = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_LABEL_MAPPING.csv'])
    pd.DataFrame(list(df.columns)).T.to_csv(path_mapping_csv, index=True)

    print('Label mapping writen to {p}.'.format(p=path_mapping_csv))

    return df


def process_N_NaIot(split: float = 0.2):
    assert 0 < split < 1

    data = get_N_BaIot()  # load each csv file, add device name and traffic type, then concat to a full dataframe

    print('Saving device_name and traffic_type info...')
    df_device_name_traffic_type = data[['device_name', 'traffic_type']]
    df_device_name_traffic_type.reset_index(drop=True, inplace=True)
    print('Done')

    print('Starting pca...')
    ndarray_pcaed = preprocess(data.drop(columns=['device_name', 'traffic_type']))
    print('Done')

    print('Renaming colmuns & adding device_name, traffic_type')
    df_pcaed = pd.DataFrame(ndarray_pcaed, columns=['dim_' + str(i) for i in range(ndarray_pcaed.shape[1])])
    df_pcaed = pd.concat([df_pcaed, df_device_name_traffic_type], axis=1)
    del df_device_name_traffic_type
    print('Done')

    print('Saving pcaed data to csv...')
    file_path_csv_paced = os.sep.join(['Dataset', 'N_BaIot', 'N_BaIot_PREPROCESSED_FULL.csv'])
    df_pcaed.to_csv(file_path_csv_paced, index=False)
    print('Done')

    pdb.set_trace()

    return train_test_split(df_pcaed, test_size=split)


def preprocess_UNSW(split: float=0.2):
    assert 0 < split < 1
    print('Starting Preprocessing')
    UNSW_data = get_UNSW()  # fetch datasets
    Xpre, ypre = preprocess(UNSW_data, 4)  # preprocess

    #Added to make sure labels and data have same length 
    ypre = pd.DataFrame(ypre)

    # Add index column; dataframe needs an index column to use pd.concat
    Xpre['id'] = Xpre.index + 1
    ypre['id'] = ypre.index + 1

    #It will just cause the csv file to have two columns with ID, need to fix that but not very important for the
    #return statement
    UNSW_data = pd.concat([Xpre, ypre], axis=1)#.drop(['id'])

    # save data-sets
    path_csv_save = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files',
                                 'a part of training and testing set', 'UNSW_NB15_PREPROCESSED.csv'])
    UNSW_data.to_csv(path_csv_save)

    Xpre = Xpre.drop(['id'], axis=1)
    ypre = ypre.drop(['id'], axis=1)

    return train_test_split(Xpre, ypre, test_size=split)
