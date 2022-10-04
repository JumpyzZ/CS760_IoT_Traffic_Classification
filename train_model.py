import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, hinge_loss, accuracy_score, confusion_matrix

from secml.adv.attacks import CAttackPoisoningSVM
from secml.ml.classifiers import CClassifierSkLearn, CClassifierSVM
from secml.array import CArray

from typing import Callable

from NN_model import *

from preprocess import *


def solver_params() -> dict:
    """
    :return: Hyper parameters for posioning attack
    """

    return {'maxIter': 100, 'tol': np.power(10.0, -5), 'eta': 0.01, 'high': 1, 'low': 0, 'h': np.power(10.0, -4)}


def evaluation_metrics(model, X_eval: Array, y_eval: Array) -> dict:
    """
    :param model:
    :param X_eval:
    :param y_eval:
    :return:
    """
    assert hasattr(model, 'predict')

    y_pred = threshold_round(model.predict(X_eval))
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1], normalize='all').ravel()

    return {'accuracy': accuracy_score(y_eval, y_pred), 'true negative': tn, 'false positive': fp,
            'false negative': fn, 'true positive': tp}


def central_difference(model: Callable, y_true: float, x: np.array, h: float) -> np.array:
    assert hasattr(model, 'loss')
    dldx = []
    loss = log_loss
    y_true = 1
    print(x)
    x = x[0]
    for i, xi in enumerate(x):
        print(i)
        x[i] += h
        #print(model.predict(x))
        L = loss(model.predict(np.array([x])), [y_true])
        print(L)
        x[i] -= 2*h
        dldx.append(np.divide(loss(threshold_round(model.predict(np.array([x]))), [y_true], labels=(0, 1)) - L, 2*h))
        #print(L, loss(threshold_round(model.predict(x)), [y_true], labels=(0, 1)))
        x[i] += h

    return np.array(dldx)


def gradient_attack(model: Callable, x: np.array, y: np.array, params: dict, gradient: Callable) -> tuple[np.array, int, int]:
    """
    :param model: model which can compute gradient (will make wrapper class later for SVM and RANDOM FOREST)
    :param gradient: a callable which defines how the algorithm handles dl_dx and dx_dw
    :param params: model parameters, like the tolerance
    :return: Feature, label, boolean; see poison_model comment
    """

    # initialise guess - possibly make Gaussian? (this is a non-biased distribution)
    iters = 0
    for i in range(params['maxIter']):
        dl_dx = central_difference(model, y, x, params['h'])
        print('DLDX', dl_dx)
        x_last = x
        iters = i

        # performs gradient ascent like method, depends on implementation of gradient function
        x = x + params['eta'] * gradient(dl_dx)
        if max(np.linalg.norm(x - x_last, 2), dl_dx) < params['tol']:
            break

    return x, y, iters


def poison_model(model: Callable, x: np.array, y: np.array, attack: str) -> Array: # assumes loss is parameterised by model
    """
    :param model: The model to be used, won't be altered
    :param attack: A string which defines the attack type
    :return: A single feature x and corresponding label y which is the result of the poisoning algorithm.
             Also True if convergence, False otherwise.
    """

    params = solver_params()
    if attack == 'maximum':
        return gradient_attack(model, x, y, params, lambda dir: dir)
    elif attack == 'fast sign':
        return gradient_attack(model, params, lambda dir: np.sign(dir))


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame,
                loss_func: Callable, attack_type: str = False) -> tuple[list, dict]:
    """
        :objective iteratively trains a SVC on dataset and poisons the dataset
        :return: model performance over iteration of posioning 
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_length = 10    # the number of sample points in any epoch
    epoch_num = 0   # the number of epochs that the model will be trained over
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
        # if next epoch has been reached train model on all data seen so far
        if np.mod(n, epoch_length) == 1 and n != 1:
            print(epoch_batch[0])
            x = np.array(epoch_batch[0])
            print(x.shape)
            model.fit(np.array(epoch_batch[0]), epoch_batch[1])

            # acquire generalisation and training loss
            loss_record[0].append(loss_func(epoch_batch[1], threshold_round(model.predict(epoch_batch[0]))))
            loss_record[1].append(loss_func(epoch_batch[3], threshold_round(model.predict(epoch_batch[2]))))

            # acquire performance metrics
            for name, value in evaluation_metrics(model, X_train, y_train).items():
                metrics[name].append(value)

            # stop after 10 epochs
            epoch_num += 1
            if epoch_num == 3:
                return loss_record, metrics

    return loss_record, metrics


def plot() -> None:
    """
    :return: trains each model using train_model in a batch style and plots the loss over epochs, displays
             the accuracy metrics
    """

    def meta_config(title: str, ylabel: str) -> None:
        # meta configuration of the plots; make them look nice
        plt.figure(figsize=(4, 4))
        plt.title(title, fontsize=25)

        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel(ylabel, fontsize=25)
        plt.xlabel('Epoch', fontsize=25)

    X_train, X_eval, y_train, y_eval = preprocess_UNSW()

    dnn = CustomNN()
    dnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy)

    models = {'SVC': svm.SVC(), 'RANDOM FOREST': RandomForestClassifier(), 'DNN': dnn}
    loss_funcs = {'SVC': hinge_loss, 'RANDOM FOREST':  tf.losses.binary_crossentropy, 'DNN': tf.losses.binary_crossentropy}

    #X_train = X_train.iloc[0:50, :]        # commented out for speed
    #y_train = y_train.iloc[0:50, :]
    #X_eval = X_eval.iloc[0:50, :]
    #y_eval = y_eval.iloc[0:50, :]

    print(np.array(y_train.iloc[0]), np.mod(y_train.iloc[0] + 1, 2))
    models['DNN'].fit(np.array(X_train), y_train)
    print(poison_model(models['DNN'], np.array([X_train.iloc[0]]), np.array(np.mod(y_train.iloc[0] + 1, 2)), 'maximum'))
    while True:
        continue
    #print(loss_grad(models['DNN'], X_train, y_train))
    #models['SVC'].fit(np.array(X_train), y_train.values.ravel())
    #models['RANDOM FOREST'].fit(np.array(X_train), y_train.values.ravel())

    print('after training metrics')
    #print('dnn', evaluation_metrics(dnn, X_eval, y_eval))
    #print('svc', evaluation_metrics(models['SVC'], X_eval, y_eval))
    #print('random forest', evaluation_metrics(models['RANDOM FOREST'], X_eval, y_eval))

    # there will be many plots : )
    recompute = True
    for n, m in enumerate(models):
        if n == 1:  # only do svm because it easier to see
            break
        # load loss history or compute and save it
        if os.path.exists(f"{m}") and not recompute:
            loss_history = np.loadtxt(f"loss--{m}")
            performance_history = np.loadtxt(f"performance--{m}") # no load atm
        else:
            loss_history, performance_history = train_model(models[m], X_train, y_train, X_eval, y_eval, loss_funcs[m])
            np.savetxt(f"loss--{m}", loss_history) # uncomment when returns

        # need to get name of loss somehow
        meta_config(f'{m} Loss', 'Loss')
        plt.plot(loss_history[0], loss_history[1], label='no posioning', linewidth=3, linestyle='solid', marker='o', markersize=9)
        plt.legend()

        for metric_name, hist in performance_history.items():
            meta_config(f'{m} {metric_name}', f'{metric_name}')
            plt.plot(hist, label=f'no posioning', linewidth=3, marker='o', linestyle='solid', markersize=9)
            plt.legend()
            np.savetxt(f"{m} {metric_name}", hist)

    plt.show()

plot()
