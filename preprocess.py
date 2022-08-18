import pdb
import os
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



def get_path_of_csv_files(top) -> list:
    """
    get all csv file path in 'top' and all its subfolers
    :param top: root dir string
    :return: ['Dataset\\N_BaIot\\Danmini_Doorbell\\benign_traffic.csv', ...] 
    """

    path_of_csv_files = []
    for path_dir_files in os.walk(top):
        for file_name in path_dir_files[2]:
            if re.search('^(?!\.)(.*?).csv', file_name) != None and file_name != 'demonstrate_structure.csv' and file_name != 'N_BaIot_PREPROCESSED_FULL.csv':
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


def preprocess(df: pd.DataFrame, PCA_PERCENTAGE = 0.99) -> pd.DataFrame:
    """
    Perform preprocessing transformations on given data
    :param df: Dataframe to be processed
    :return: Dataframe of transformed data 
    """

    array1 = StandardScaler().fit_transform(df)    # standardize data
    array2 = PCA(n_components = PCA_PERCENTAGE).fit_transform(array1) # perform PCA on standardized data
    array3 = array2 + np.random.normal(0, 0.1, size=array2) # add gaussian white noise 

    return pd.DataFrame(array3)


def get_N_BaIot() -> pd.DataFrame:
    """
    :return: a dataframe
    """

    count = 1
    dataset_dir = 'Dataset' + os.sep + 'N_BaIot'
    file_list = [get_file_type(file_path) for file_path in get_path_of_csv_files(dataset_dir)]
    data = pd.DataFrame()
    for file in file_list:
        print('(loading {c}/{f})'.format(c = count, f = len(file_list)), file['file_path'])
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
    test_path = os.sep.join(['Dataset','UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_testing-set.csv'])

    # fetch data 
    df = pd.concat([pd.read_csv(train_path, header=0), pd.read_csv(test_path, header=0)], axis=0, ignore_index=False)

    # convert non-numerical entries to numerical 
    mapping_list = []
    string_cloumns = df.dtypes.loc[df.dtypes == 'object'].index.to_list()
    for string_cloumn in string_cloumns:
        le = LabelEncoder()
        df[string_cloumn] = le.fit_transform(df[string_cloumn])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_list.append(le_name_mapping)
    
    path_mapping_csv = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_LABEL_MAPPING.csv'])
    pd.DataFrame(mapping_list).T.to_csv(path_mapping_csv, index = True)
    print('Label mapping writen to {p}.'.format(p = path_mapping_csv))
    
    return pd.DataFrame(preprocess(df, 0.9))


def process_N_NaIot(split: float = 0.2):
    assert 0 < split < 1

    data = get_N_BaIot() # load each csv file, add device name and traffic type, then concat to a full dataframe

    print('Saving device_name and traffic_type info...')
    df_device_name_traffic_type = data[['device_name', 'traffic_type']]
    df_device_name_traffic_type.reset_index(drop = True, inplace = True)
    print('Done')

    print('Starting pca...')
    ndarray_pcaed = preprocess(data.drop(columns=['device_name', 'traffic_type']))
    print('Done')

    print('Renaming colmuns & adding device_name, traffic_type')
    df_pcaed = pd.DataFrame(ndarray_pcaed, columns = ['dim_'+str(i) for i in range(ndarray_pcaed.shape[1])])
    df_pcaed = pd.concat([df_pcaed, df_device_name_traffic_type], axis = 1)
    del df_device_name_traffic_type
    print('Done')

    print('Saving pcaed data to csv...')
    file_path_csv_paced = os.sep.join(['Dataset','N_BaIot','N_BaIot_PREPROCESSED_FULL.csv'])
    df_pcaed.to_csv(file_path_csv_paced, index = False)
    print('Done')

    pdb.set_trace()

    return train_test_split(df_pcaed, test_size=split)


def preprocess_UNSW(split: float = 0.2):
    assert 0 < split < 1

    UNSW_data = get_UNSW() # fetch datasets

    print('Starting PCA')
    UNSW_PCA = pd.DataFrame(preprocess(UNSW_data, 0.9)) # PCA
    print('Done.')

    print('Saving data to csv') # save data-sets
    path_csv_save = os.sep.join(['Dataset','UNSW-NB15 - CSV Files','a part of training and testing set', 'UNSW_NB15_PREPROCESSED_training-set.csv'])
    UNSW_PCA.to_csv(path_csv_save)
    print('Done.')

    return train_test_split(UNSW_PCA, test_size=split)
