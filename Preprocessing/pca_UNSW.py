###  This script should run under .../CS760_IoT_Traffic_Classification/  ###
###  due to dataset size, 23GB+ ram is required, on my 16GB pc, when saving pcaed data to csv: numpy.core._exceptions.MemoryError: Unable to allocate 6.05 GiB for an array with shape (7062606, 115) and data type float64

import pdb
import os
from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pca_N_BaIot import standardize_and_pca


def get_UNSW() -> pd.DataFrame:
    """
    :return: The UNSW dataset
    """
    train_path = os.sep.join(['Dataset', 'UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_training-set.csv'])
    test_path = os.sep.join(['Dataset','UNSW-NB15 - CSV Files', 'a part of training and testing set', 'UNSW_NB15_testing-set.csv'])
    df = pd.concat([pd.read_csv(train_path, header=0), pd.read_csv(test_path, header=0)], axis=0, ignore_index=False)

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
    
    return df


if __name__ == '__main__':

    # fetch datasets
    # NbaIoT_data = get_NBaIot() there is a bug somewhere here for me
    UNSW_data = get_UNSW()

    # PCA
    print('Starting PCA')
    UNSW_PCA = pd.DataFrame(standardize_and_pca(UNSW_data, 0.9))
    print('Done.')

    # save data-sets
    print('Saving data to csv')
    path_csv_save = os.sep.join(['Dataset','UNSW-NB15 - CSV Files','a part of training and testing set', 'UNSW_NB15_PREPROCESSED_training-set.csv'])
    UNSW_PCA.to_csv(path_csv_save)
    print('Done.')

    pdb.set_trace()
