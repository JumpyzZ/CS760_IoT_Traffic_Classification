import matplotlib.pyplot as plt

from CNN import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, hinge_loss, accuracy_score, confusion_matrix

from typing import Callable, List, Any, Optional

from NN_model import *

from preprocess import *


def solver_params() -> dict:
    """
    :return: Hyper parameters for posioning attack
    """

    return {'maxIter': 100, 'tol': np.power(10.0, -10), 'eta': 0.5, 'high': 1, 'low': 0, 'h': np.power(10.0, -4)}


def evaluation_metrics(model, X_eval: Array, y_eval: Array) -> dict:
    """
    :param model:
    :param X_eval:
    :param y_eval:
    :return:
    """
    assert hasattr(model, 'predict')

    y_pred = myround(model.predict(X_eval))
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1], normalize='all').ravel()

    return {'accuracy': accuracy_score(y_eval, y_pred), 'true negative': tn, 'false positive': fp,
            'false negative': fn, 'true positive': tp}


def loss_central_difference(model: Callable, x: np.array, h: float, y: int, loss: Callable) -> np.array:
    dldx = []
    for i, xi in enumerate(x):
        x[i] = xi - h
        L0 = loss(y_pred=model.predict(np.array([x])), y_true=[y], labels=(0, 1))
        x[i] = xi + h
        L1 = loss(y_pred=model.predict(np.array([x])), y_true=[y], labels=(0, 1))
        dldx.append((L1 - L0)/2*h)
        x[i] = xi

    return np.array(dldx)


def model_central_difference(model: Callable, x: np.array, h: float, *args):
    dfdx = []
    for i, xi in enumerate(x):
        x[i] = xi - h
        L0 = model.predict(np.array([x]))
        x[i] = xi + h
        L1 = model.predict(np.array([x]))
        dfdx.append((L1[0][0] - L0[0][0])/2*h)
        x[i] = xi

    return np.array(dfdx)


def gradient_attack(model: Callable, loss: Callable, gradient: Callable,
                    x: np.array, y: int, params: dict,
                    stop: Union[Callable, bool], dx: Callable) -> tuple[np.array, int]:
    """
    :param y:
    :param loss:
    :param x:
    :param stop:
    :param derivative:
    :param model: model which can compute gradient (will make wrapper class later for SVM and RANDOM FOREST)
    :param gradient:
    :param params: model parameters, like the tolerance
    :return: Feature, label, boolean; see poison_model comment
    """

    iters = 0
    f0 = model.predict(np.array([x]))
    f1 = f0
    for i in range(params['maxIter']):
        derivative = dx(model, x, params['h'], y, loss)

        x_last = x
        iters = i

        # performs gradient ascent like method, depends on implementation of gradient function
        x = x + params['eta'] * gradient(f1, derivative, y)[0]
        f1 = model.predict(np.array([x]))
        print(f1, f0, y, myround(f0))
        if max([np.linalg.norm(x - x_last, 2)]) < params['tol'] or stop(f0, f1):
            iters = i + 1
            break

    return x, iters


def poison_model(model: Callable, loss: Callable, data: tuple, attack: str, num_data: int) -> tuple:
    """
    :param model: The model to be used, won't be altered
    :param attack: A string which defines the attack type
    :return: A single feature x and corresponding label y which is the result of the poisoning algorithm.
             Also True if convergence, False otherwise.
    """

    assert hasattr(model, 'fit') and len(data) == 4

    if attack == 'FGSM':
        func, stop = lambda val, dir, y: np.sign(dir), lambda y1, y2: False
        dx = loss_central_difference
    elif attack == 'DeepFool':
        func, stop = lambda val, dir, y: np.power(-1, y + 1) * dir * (val/np.linalg.norm(dir, 2)), lambda y1, y2: myround(y1) != myround(y2)
        dx = model_central_difference
    elif attack == 'Norm':
        func, stop = lambda val, dir, y: dir / np.linalg.norm(dir, 2), lambda y1, y2: False
        dx = loss_central_difference
    else:
        raise ValueError("Invalid attack type")

    params = solver_params()
    X_train, y_train, X_eval, y_eval = data
    X_po, y_po, loss_history, iters = [], [], [], []
    for _ in range(num_data):
        k = np.random.randint(0, X_train.shape[0])
        x = np.array([X_train.iloc[k]][0])
        yp = np.mod(y_train.iloc[k] + 1, 2)[0]

        xp, i = gradient_attack(model, loss, func, x, yp, params, stop, dx=dx)

        model.fit(np.array([xp]), np.array([yp]))

        test_loss = loss(y_pred=model.predict(np.array([xp])), y_true=np.array([yp]), labels=(0, 1))
        #train_loss = loss(y_pred=model.predict(X_eval), y_true=y_eval, labels=(0, 1))

        loss_history.append(test_loss)
        X_po.append(x)
        y_po.append(yp)
        iters.append(i)

    return X_po, y_po, loss_history, iters


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame,
                loss_func: Callable) -> tuple[list, dict]:
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
            x = np.array(epoch_batch[0])
            model.fit(np.array(epoch_batch[0]), epoch_batch[1])

            # acquire generalisation and training loss
            loss_record[0].append(loss_func(epoch_batch[1], myround(model.predict(epoch_batch[0]))))
            loss_record[1].append(loss_func(epoch_batch[3], myround(model.predict(epoch_batch[2]))))

            # acquire performance metrics
            for name, value in evaluation_metrics(model, epoch_batch[0], epoch_batch[1]).items():
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
    cnn = CNN_Model(X_train, y_train)
    dnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy)
    cnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy)

    while True:
        continue

    models = {'DNN': dnn, 'cnn': cnn}
    loss_funcs = {'DNN': log_loss, 'CNN': log_loss}

    X_train = X_train.iloc[0:50, :]        # commented out for speed
    y_train = y_train.iloc[0:50, :]
    X_eval = X_eval.iloc[0:50, :]
    y_eval = y_eval.iloc[0:50, :]

    models['DNN'].fit(X_train, y_train)
    #models['CNN'].fit(X_train, y_train)

    # model poisoning
    #fX, fy, fl_hist, fi = poison_model(models['DNN'], loss_funcs['DNN'], (X_train, y_train, X_eval, y_eval), 'FGSM', 1)
    dX, dy, dl_hist, di = poison_model(models['DNN'], loss_funcs['DNN'], (X_train, y_train, X_eval, y_eval), 'DeepFool', 1)

    print(di)

    plt.plot([np.linalg.norm(x, 2) for x in dX], linewidth=3, linestyle='solid', marker='o', markersize=9)
    plt.ylabel('||x||')
    plt.xlabel(r'k')
    plt.xticks(ticks=range(len(dX)))
    plt.show()

    print('after training metrics')
    #print('dnn', evaluation_metrics(dnn, X_eval, y_eval))
    #print('svc', evaluation_metrics(models['SVC'], X_eval, y_eval))
    #print('random forest', evaluation_metrics(models['RANDOM FOREST'], X_eval, y_eval))

    recompute = True
    for n, m in enumerate(models):
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
