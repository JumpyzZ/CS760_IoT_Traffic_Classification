import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import log_loss, hinge_loss
from NN_model import *

from preprocess import * 


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[list, list]:
    """
        :objective: iteratively trains a SVC on dataset and posions the dataset
        :return: model performance over iteration of posioning 
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_length = 100 #int(np.ceil(len(X_train)) / 100)
    epoch_batch = [[], np.array([]), [], []]
    cross_entropy = []
    hinge = []
    count = 0

    for n, xtr, ytr, xte, yte in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()):
        epoch_batch[0].append(xtr)
        epoch_batch[1] = np.append(epoch_batch[1], ytr, axis=0)
        epoch_batch[2].append(xte)
        epoch_batch[3] = np.append(epoch_batch[3], yte, axis=0)
        if np.mod(n, epoch_length) == 1 and n != 1:
            model.fit(epoch_batch[0], epoch_batch[1])
            predictions = model.predict(epoch_batch[2])
            cross_entropy.append(log_loss(predictions, epoch_batch[3]))
            hinge.append(hinge_loss(predictions, epoch_batch[3]))
            count += 1
            if count == 10:
                return cross_entropy, hinge

    return cross_entropy, hinge


def plot():
    entropy_filename = 'SVM - binary cross entropy'
    hinge_filename = 'SVM - hinge loss'

    recompute = False
    if os.path.exists(entropy_filename) and os.path.exists(hinge_filename) and not recompute:
        cross = np.loadtxt(entropy_filename)
        hinge = np.loadtxt(hinge_filename)
    else:
        svc = svm.SVC(C=1, kernel='rbf')
        X_train, X_test, y_train, y_test = preprocess_UNSW()
        cross, hinge = train_model(svc, X_train, y_train, X_test, y_test)

    np.savetxt('SVM - binary cross entropy', cross)
    np.savetxt('SVM - hinge loss', hinge)

    fig = plt.figure()
    plt.title("Loss Of SVM", fontsize=25)

    #plt.plot(cross, label='binary cross entropy', linewidth=2)
    plt.plot(hinge, label='hinge loss', linewidth=3, marker='o', linestyle='dashed', markersize=9)

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Loss', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)

    plt.show()

plot()