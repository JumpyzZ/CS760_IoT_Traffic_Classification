import pandas as pd
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import itertools
import os
import tensorflow as tf

from sklearn.svm import SVC
from LSTM_new import SequeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from typing import Union
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, confusion_matrix, plot_roc_curve
from typing import Callable, List, Any, Optional
from numpy import loadtxt
from keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn import datasets
from keras.layers import *
from keras import Model
from keras.utils import plot_model,to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras.layers import Dropout,SimpleRNN

# CNN
def CNN_Model(fea_cnt,numb):
    K.clear_session() # clear memory
    inputs = Input(shape=(fea_cnt,), dtype='float32')
    embds = Dense(64 ,activation='selu')(inputs)
    embds = Reshape((8,8,1))(embds)
    embds = Conv2D(1,(4,4),strides=(1,1),padding="valid")(embds)
    embds = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')(embds)
    embds = Flatten()(embds)
    embds = Dense(64,activation='selu')(embds)
    embds = Dense(32,activation='selu')(embds)
    outputs = Dense(1,activation='sigmoid')(embds)

    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model

# DNN
Numeric = Union[float, int]
Array = Union[np.array, list]
class CustomNN(tf.keras.Model):
    def __init__(self, n: int = 10, weight_init=None) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(n + 10, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(n + 10, activation=tf.nn.selu)
        self.dense3 = tf.keras.layers.Dense(5, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs: np.array) -> np.array:
        features1 = self.dense1(inputs)
        features2 = self.dense2(features1)
        features3 = self.dense3(features2)
        features4 = self.dense4(features3)

        return features4



# RNN
def RNN_base():
    model = Sequential()
    #  1st RNN layer and Dropout regularization
    # model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=(x_train_.shape[1], 1)))
    X = np.array(X_train).reshape([-1, X_train.shape[1], 1])
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True, input_shape=X))
    model.add(Dropout(0.2))
    # 2nd RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    # 3rd RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    # 4th RNN layer and Dropout regularization
    model.add(SimpleRNN(units=50))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(1, activation='sigmoid'))

    return model

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


def preprocess_UNSW(split: float=0.2):
    assert 0 < split < 1
    print('Starting Preprocessing')
    UNSW_data = get_UNSW()  # fetch datasets
    Xpre, ypre = preprocess(UNSW_data, 10)  # preprocess

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


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame,
                loss_func: Callable, epoch_num: int, epoch_length: int) -> tuple[list, dict]:
    """
        :objective iteratively trains a SVC on dataset and poisons the dataset
        :return: model performance over iteration of posioning
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_batch = [[], np.array([]), [], np.array([])]  # X_train. y_train, X_test, y_test
    loss_record = [[], []]  # generalisation and training loss arrays
    metrics = {'accuracy': [], 'true negative': [], 'false positive': [], 'false negative': [], 'true positive': []}    # performance metrics

    append = lambda a, x: [x] if len(a) == 0 else np.append(a, [x], axis=0)

    for n, xtr, ytr, xte, yte in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()):
        # add sample to batch
        epoch_batch[0] = append(epoch_batch[0], xtr)
        epoch_batch[1] = np.append(epoch_batch[1], ytr, axis=0)
        epoch_batch[2] = append(epoch_batch[2], xte)
        epoch_batch[3] = np.append(epoch_batch[3], yte, axis=0)

        if epoch_num == n:
            return loss_record, metrics

        # if next epoch has been reached train model on all data seen so far
        if np.mod(n, epoch_length) == 0:
            print("...")
            model.fit(np.array(epoch_batch[0]), epoch_batch[1])

            # acquire generalisation and training loss
            loss_record[0].append(loss_func(epoch_batch[1], myround(model.predict(epoch_batch[0]))))
            loss_record[1].append(loss_func(epoch_batch[3], myround(model.predict(epoch_batch[2]))))

            # acquire performance metrics
            for name, value in evaluation_metrics(model, epoch_batch[0], epoch_batch[1]).items():
                metrics[name].append(value)

    return loss_record, metrics

def preprocess(data: pd.DataFrame, num_features: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
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



def myround(predictions):
    return np.array([1 if t > 0.5 else 0 for t in predictions])

def evaluation_metrics(model, X_eval, y_eval):
    """
    :param model:
    :param X_eval:
    :param y_eval:
    :return:
    """
    assert hasattr(model, 'predict')

    y_pred = myround(model.predict(X_eval))
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1], normalize='all').ravel()

    return {'true negative': tn, 'false positive': fp, 'false negative': fn, 'true positive': tp}


def performance_plot() -> None:
    """
    :return: trains each model using train_model in a batch style and plots the loss over epochs, displays
             the accuracy metrics
    """

    models['DNN'].fit(X_train, y_train)
    models['CNN'].fit(X_train, y_train)
    models['LSTM'].fit(X_train, y_train)
    models['RNN'].fit(np.array(X_train).reshape([-1, X_train.shape[1], 1]), y_train)
    models['RF'].fit(X_train.values, y_train.values.ravel())
    models['SVM'].fit(X_train, y_train.values.ravel())

    print('after training metrics')
    dnn_metrics = evaluation_metrics(dnn, X_eval, y_eval)
    cnn_metrics = evaluation_metrics(cnn, X_eval, y_eval)
    lstm_metrics = evaluation_metrics(lstm, X_eval, y_eval)
    rnn_metrics = evaluation_metrics(rnn, np.array(X_eval).reshape([-1, X_eval.shape[1], 1]), y_eval)
    svm_metrics = evaluation_metrics(models['SVM'], X_eval, y_eval)
    rf_metrics = evaluation_metrics(models['RF'], X_eval, y_eval)

    print('dnn', dnn_metrics)
    print('cnn', cnn_metrics)
    print('lstm', lstm_metrics)
    print('rnn', rnn_metrics)
    print('svm', svm_metrics)
    print('rf', rf_metrics)

    plt.figure()
    fig, ax = plt.subplots()
    x = np.arange(len(dnn_metrics))
    width = 0.15  # the width of the bars

    rects1 = ax.bar(x - width, list(dnn_metrics.values()), width, label='DNN')
    rects2 = ax.bar(x + width , list(cnn_metrics.values()), width, label='CNN')
    rects3 = ax.bar(x, list(lstm_metrics.values()), width, label='LSTM')
    rects4 = ax.bar(x - 2 * width, list(rnn_metrics.values()), width, label='RNN')
    rects5 = ax.bar(x + 2 * width, list(svm_metrics.values()), width=width, label='SVM')
    rects6 = ax.bar(x - 3 * width, list(rf_metrics.values()), width, label='RF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=15)
    ax.set_title('Model Performance metrics', fontsize=15)
    ax.set_xticks(x, list(dnn_metrics.keys()), fontsize=13, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)

    fig.tight_layout()

    #plt.figure()
    #m = tf.keras.metrics.AUC(curve='ROC')
    #m.update_state(y_eval, models['DNN'].predict(X_eval))

    #plot_roc_curve(dnn, X_eval, y_eval)
    plt.savefig('./boxplot.eps', dpi=300)
    plt.show()


def Draw_ROC(model1,model2,model3, model4, model5, model6):

    fpr_DNN,tpr_DNN,thresholds=roc_curve(np.array(y_eval),model1.predict(X_eval))
    roc_auc_DNN=auc(fpr_DNN,tpr_DNN)

    fpr_CNN,tpr_CNN,thresholds=roc_curve(np.array(y_eval),model2.predict(X_eval))
    roc_auc_CNN=auc(fpr_CNN,tpr_CNN)

    fpr_LSTM,tpr_LSTM,thresholds=roc_curve(np.array(y_eval),model3.predict(X_eval)
    roc_auc_LSTM=auc(fpr_LSTM,tpr_LSTM)

    fpr_RNN,tpr_RNN,thresholds=roc_curve(np.array(y_eval),model4.predict(np.array(X_eval).reshape([-1, X_eval.shape[1], 1])))
    roc_auc_RNN=auc(fpr_RNN,tpr_RNN)

    fpr_RF,tpr_RF,thresholds=roc_curve(np.array(y_eval),model5.predict(np.array(X_eval)))
    roc_auc_RF=auc(fpr_RF,tpr_RF)

    fpr_SVM,tpr_SVM,thresholds=roc_curve(np.array(y_eval),model6.predict(np.array(X_eval)))
    roc_auc_SVM=auc(fpr_SVM,tpr_SVM)


    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))

    plt.plot(fpr_DNN,tpr_DNN,'purple',lw=lw,label='DNN_AUC = %0.2f'% roc_auc_DNN)

    plt.plot(fpr_CNN,tpr_CNN,color='darkorange', lw=lw, label='CNN_AUC = %0.2f'% roc_auc_CNN)

    plt.plot(fpr_LSTM,tpr_LSTM,color='red', lw=lw, label='LSTM_AUC = %0.2f'% roc_auc_LSTM)

    plt.plot(fpr_RNN,tpr_RNN,color='green', lw=lw, label='RNN_AUC = %0.2f'% roc_auc_RNN)

    plt.plot(fpr_RF,tpr_RF,color='black', lw=lw, label='RF_AUC = %0.2f'% roc_auc_RF)

    plt.plot(fpr_SVM,tpr_SVM,color='brown', lw=lw, label='SVM_AUC = %0.2f'% roc_auc_SVM)

    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.title('ROC Curve')
    plt.savefig('./ROCcurve.eps', dpi=300)
    plt.show()


X_train, X_eval, y_train, y_eval = preprocess_UNSW()

n = X_train.shape[1]

dnn = CustomNN(n, tf.keras.initializers.he_uniform)

cnn = CNN_Model(n, 2)

lstm = SequeClassifier(128)
lstm.build_model()

rnn = RNN_base()

rf = RandomForestClassifier()
svc = SVC(kernel='rbf', cache_size=100, class_weight={0: 0.5, 1: 1}, probability=True, verbose=10)


dnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy, metrics=[tf.metrics.TruePositives()])
cnn.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
lstm.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
rnn.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])

models = {'DNN': dnn, 'CNN': cnn, 'LSTM':lstm, 'RNN':rnn, "SVM":svc, "RF":rf}
loss_funcs = {'DNN': log_loss, 'CNN': log_loss, 'LSTM':log_loss, 'RNN':log_loss,  'SVM':log_loss, 'RF':log_loss}

#X_train = X_train.iloc[0:50, :]        # for speed
#y_train = y_train.iloc[0:50, :]
#X_eval = X_eval.iloc[0:50, :]
#y_eval = y_eval.iloc[0:50, :]

performance_plot()

Draw_ROC(dnn,cnn,lstm, rnn, rf, svc)
