import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import log_loss, hinge_loss, accuracy_score
from NN_model import *

from preprocess import * 


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame) -> list:
    """
        :objective: iteratively trains a SVC on dataset and posions the dataset
        :return: model performance over iteration of posioning 
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_length = 200 #int(np.ceil(len(X_train)) / 100)
    epoch_batch = [[], np.array([]), [], np.array([])]
    hinge = []
    count = 0

    for n, xtr, ytr, xte, yte in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()):
        epoch_batch[0].append(xtr)
        epoch_batch[1] = np.append(epoch_batch[1], ytr, axis=0)
        epoch_batch[2].append(xte)
        epoch_batch[3] = np.append(epoch_batch[3], yte, axis=0)
        if np.mod(n, epoch_length) == 1 and n != 1:
            model.fit(np.array(epoch_batch[0]), epoch_batch[1])
            predictions = np.array([1 if t >= 0.5 else 0 for t in model.predict(np.array(epoch_batch[2]))])
            hinge.append(hinge_loss(predictions, epoch_batch[3]))
            count += 1
            if count == 10:
                return hinge

    return hinge


def plot():
    hinge_filename = ' - hinge loss'

    svc = svm.SVC(C=1, kernel='rbf')
    dnn = CustomNN()
    dnn.compile(optimizer='Adam', loss='mse')

    models = {'SVC': svc, 'DNN': dnn}
    colors = ['tab:']
    hinge = [[], []]

    X_train, X_test, y_train, y_test = preprocess_UNSW()

    dnn.fit(X_train, y_train)
    svc.fit(np.array(X_train), np.ravel(y_train))
    print('starting')
    print('svc: ', accuracy_score(np.array(y_test), np.array(svc.predict(np.array(X_test)))))
    print('nn: ', accuracy_score(np.array(y_test), dnn.predict(np.array(X_test))))

    recompute = False
    fig = plt.figure()
    for n, m in enumerate(models):
        if os.path.exists(f"{m} - {hinge_filename}") and not recompute:
            hinge[n].append(np.loadtxt(f"{m}{hinge_filename}"))
        else:
            hinge[n] = np.append(hinge[n], train_model(models[m], X_train, y_train, X_test, y_test), axis=0)
            np.savetxt(f'{m}{hinge_filename}', hinge[n])

        hinge[1] = np.loadtxt('DNN_-_hinge_loss')
        hinge[0] = np.loadtxt('SVC_-_hinge_loss')
        plt.plot(hinge[n], label=f'{m} hinge loss', linewidth=3, linestyle='dashed', marker='o', markersize=9)

    plt.title("Model Loss Evaluation", fontsize=25)

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Loss', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)

    plt.show()


plot()
