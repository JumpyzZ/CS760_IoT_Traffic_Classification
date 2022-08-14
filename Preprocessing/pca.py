###  This script should run under .../CS760_IoT_Traffic_Classification/  ###
###  due to dataset size, 23GB+ ram is required, on my 16GB pc, when saving pcaed data to csv: numpy.core._exceptions.MemoryError: Unable to allocate 6.05 GiB for an array with shape (7062606, 115) and data type float64

import pdb
import os
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# get all csv file path releative to .../CS760_IoT_Traffic_Classification/
# return: ['Dataset\N_BaIot\Danmini_Doorbell\benign_traffic.csv', ...] 
def get_path_of_csv_files(top):
    path_of_csv_files = []
    for path_dir_files in os.walk(top):
        for file_name in path_dir_files[2]:
            if re.search('^(?!\.)(.*?).csv', file_name) != None and file_name != 'demonstrate_structure.csv' and file_name != 'N_BaIot_PCAed.csv':
                path_of_csv_files.append(path_dir_files[0] + os.sep + file_name)
    return path_of_csv_files


# get device name and traffic type based on file path
# return: {'file_path': 'Dataset\\N_BaIot\\SimpleHome_XCS7_1003_WHT_Security_Camera\\mirai_attacks\\udpplain.csv',
#           'device_name': 'SimpleHome_XCS7_1003_WHT_Security_Camera',
#           'traffic_type': 'mirai_attacks-udpplain'}
def get_file_type(file_path):
    device_name = file_path.split(os.sep)[2]
    traffic_type = ''
    if file_path.count(os.sep) == 3 and file_path.split(os.sep)[-1] == 'benign_traffic.csv':
        traffic_type = 'benign_traffic'
    if file_path.count(os.sep) == 4:
        traffic_type = file_path.split(os.sep)[3] + '-' + file_path.split(os.sep)[4].replace('.csv', '')
    return {'file_path': file_path,
            'device_name': device_name,
            'traffic_type': traffic_type}


# perform standardize and pca on given data
# return: a ndarray
def standardize_and_pca(df, PCA_PERCENTAGE = 0.99):
    # standardize data
    scaler = StandardScaler()
    ndarray_scaled = scaler.fit_transform(df)
    # perform PCA on standardized data
    pca = PCA(n_components = PCA_PERCENTAGE)
    ndarray_scaled_pca = pca.fit_transform(ndarray_scaled)
    return ndarray_scaled_pca


if __name__ == '__main__':
    # load each csv file, add device name and traffic type, then concat to a full dataframe
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

    print('Saving device_name and traffic_type info...')
    df_device_name_traffic_type = data[['device_name', 'traffic_type']]
    df_device_name_traffic_type.reset_index(drop = True, inplace = True)
    print('Done')

    print('Starting pca...')
    ndarray_pcaed = standardize_and_pca(data.drop(columns=['device_name', 'traffic_type']))
    print('Done')

    print('Renaming colmuns & adding device_name, traffic_type')
    df_pcaed = pd.DataFrame(ndarray_pcaed, columns = ['dim_'+str(i) for i in range(ndarray_pcaed.shape[1])])
    df_pcaed = pd.concat([df_pcaed, df_device_name_traffic_type], axis = 1)
    del df_device_name_traffic_type
    print('Done')

    print('Saving pcaed data to csv...')
    file_path_csv_paced = os.sep.join(['Dataset','N_BaIot','N_BaIot_PCAed.csv'])
    df_pcaed.to_csv(file_path_csv_paced, index = False)
    print('Done')
    pdb.set_trace()
