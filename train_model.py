import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, hinge_loss, accuracy_score

from secml.adv.attacks import CAttackPoisoningSVM
from secml.array import CArray

from NN_model import *

from preprocess import * 


def solver_params() -> dict:
    """
    :return: Parameter array for CAttackPoisoning[model]
    """
    return {'eta': 0.05, 'eta_min': 0.05, 'eta_max': None, 'max_iter': 100, 'eps': 1e-6}


def poison_svm(svm: svm.SVC, X_train: pd.DataFrame, y_train: pd.DataFrame,
               X_test: pd.DataFrame, y_test: pd.DataFrame, num_points: int = 1):
    """
    :param svm:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param num_points:
    :return:
    """

    # obtain training and validation sets
    training_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # convert dataframes to CArray
    X_train, y_train, X_test, y_test = CArray(X_train.values), CArray(y_train.values), CArray(X_test.values), CArray(y_test.values)
    training_data, test_data = CArray(training_data.values), CArray(test_data.values)

    svm_attack = CAttackPoisoningSVM(classifier=svm,
                                     training_data=training_data,
                                     val=test_data,
                                     solver_params=solver_params(),
                                     random_seed=np.random.randint(1, 400, 1))

    svm_attack.n_points = num_points

    attack_ypred, attack_score, attack_ds, f_opt = svm_attack.run(X_train, y_train)


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame) -> list:
    """
        :objective: iteratively trains a SVC on dataset and poisons the dataset
        :return: model performance over iteration of posioning 
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_length = 1 #int(np.ceil(len(X_train)) / 100)
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
            predictions = np.array(threshold(model.predict(np.array(epoch_batch[2]))))
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
    random_forest = RandomForestClassifier()

    models = {'SVC': svc, 'DNN': dnn}
    hinge = [[], [], []]

    X_train, X_test, y_train, y_test = preprocess_UNSW()

    X_train = X_train.iloc[0:20, :]
    y_train = y_train.iloc[0:20, :]
    X_test = X_test.iloc[0:20, :]
    y_test = y_test.iloc[0:20, :]

    dnn.fit(X_train, y_train)
    svc.fit(np.array(X_train), np.ravel(y_train))

    print('svc: ', accuracy_score(np.array(y_test), np.array(svc.predict(np.array(X_test)))))
    print('nn: ', accuracy_score(np.array(y_test), threshold(dnn.predict(np.array(X_test)))))

    recompute = False
    fig = plt.figure()
    for n, m in enumerate(models):
        if os.path.exists(f"{m} - {hinge_filename}") and not recompute:
            hinge[n].append(np.loadtxt(f"{m}{hinge_filename}"))
        else:
            hinge[n] = np.append(hinge[n], train_model(models[m], X_train, y_train, X_test, y_test), axis=0)
            np.savetxt(f'{m}{hinge_filename}', hinge[n])

        plt.plot(hinge[n], label=f'{m} hinge loss', linewidth=3, linestyle='dashed', marker='o', markersize=9)

    plt.title("Model Loss Evaluation", fontsize=25)

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Loss', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)

    plt.show()


plot()
