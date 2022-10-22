import matplotlib.pyplot as plt
import pandas as pd

from CNN_new import CNN_Model
from LSTM import LSTM_model, lstmClass
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, confusion_matrix, plot_roc_curve

from typing import Callable, List, Any, Optional

from NN_model import *

from preprocess import *


def solver_params() -> dict:
    """
    :return: Hyper parameters for posioning attack
    """

    return {'maxIter': 100, 'tol': np.power(10.0, -10), 'eta': 0.1, 'high': 1, 'low': 0, 'h': np.power(10.0, -4)}


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

    return {'true negative': tn, 'false positive': fp, 'false negative': fn, 'true positive': tp}


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
    f0 = model.predict(np.array([x]), verbose=0)
    f1 = f0
    for i in range(params['maxIter']):
        derivative = dx(model, x, params['h'], y, loss)

        x_last = x
        iters = i

        # performs gradient ascent like method, depends on implementation of gradient function
        x = x + params['eta'] * gradient(f1, derivative, y)[0]
        f1 = model.predict(np.array([x]), verbose=0)
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
        func = lambda val, dir, y: np.power(-1, y + 1) * dir * (val/np.linalg.norm(dir, 2))
        stop = lambda y1, y2: myround(y1) != myround(y2)
        dx = model_central_difference
    elif attack == 'Norm':
        func, stop = lambda val, dir, y: dir / np.linalg.norm(dir, 2), lambda y1, y2: False
        dx = loss_central_difference
    else:
        raise ValueError("Invalid attack type")

    params = solver_params()
    X_train, y_train, X_eval, y_eval = data
    X_po, y_po, model_confidences = [], [], [[], []]
    for _ in range(num_data):
        for _ in range(100):
            k = np.random.randint(0, X_train.shape[0])
            x = np.array([X_train.iloc[k]][0])
            yp = np.mod(y_train.iloc[k] + 1, 2)[0]
            if myround(model.predict(np.array([x])))[0] == y_train.iloc[k].values[0]:
                break

        model_confidences[0].append(model.predict(np.array([x]), verbose=0)[0][0])
        xp, i = gradient_attack(model, loss, func, x, yp, params, stop, dx=dx)
        model_confidences[1].append(model.predict(np.array([xp]), verbose=0)[0][0])

        X_po.append(xp)
        y_po.append(yp)

        model.fit(np.array([xp]), np.array([yp]), verbose=0)

        sample_loss = loss(y_pred=model.predict(np.array([xp]), verbose=0), y_true=np.array([yp]), labels=(0, 1))
        test_loss = loss(y_pred=model.predict(X_eval, verbose=0), y_true=y_eval, labels=(0, 1))



    # model confidence in sample before/after
    # sample itself

    return np.array(X_po), np.array(y_po), model_confidences


def train_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame,
                loss_func: Callable, epoch_num: int, epoch_length: int) -> tuple[list, dict]:
    """
        :objective iteratively trains a SVC on dataset and poisons the dataset
        :return: model performance over iteration of posioning 
    """

    assert hasattr(model, 'fit') and hasattr(model, 'predict')

    epoch_batch = [np.array([[]]), np.array([]), np.array([[]]), np.array([])]  # batches
    loss_record = [[], []]  # training and generalisation loss arrays
    metrics = {'accuracy': [], 'true negative': [], 'false positive': [], 'false negative': [], 'true positive': []}    # performance metrics

    I = lambda a: 1 if a.shape[1] == 0 else 0
    for n, xtr, ytr, xte, yte in zip(range(len(X_train)), X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()):
        # add sample to batch
        epoch_batch[0] = np.append(epoch_batch[0], [xtr], axis=I(epoch_batch[0]))
        epoch_batch[1] = np.append(epoch_batch[1], ytr, axis=0)
        epoch_batch[2] = np.append(epoch_batch[2], [xte], axis=I(epoch_batch[2]))
        epoch_batch[3] = np.append(epoch_batch[3], yte, axis=0)

        if (epoch_num-1)*epoch_length == n:
            return loss_record, metrics

        # if next epoch has been reached train model on all data seen so far
        if np.mod(n, epoch_length-1) == 0:
            #print(epoch_batch[0], epoch_batch[1])
            model.fit(epoch_batch[0], epoch_batch[1])

            # acquire training and generalisation loss
            loss_record[0].append(loss_func(epoch_batch[1], model.predict(epoch_batch[0]), labels=(0, 1)))
            loss_record[1].append(loss_func(epoch_batch[3], model.predict(epoch_batch[2]), labels=(0, 1)))

            # acquire performance metrics
            for name, value in evaluation_metrics(model, epoch_batch[0], epoch_batch[1]).items():
                metrics[name].append(value)

    return loss_record, metrics


def defensive_mechanism(X_train: pd.DataFrame, X_po: pd.DataFrame):
    mu = np.mean(X_train, axis=1)
    std = np.std(X_train, axis=1)


def poisoning_plot():
    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    for name, model in models.items():
        fX, fy, fl_hist, fi = poison_model(model, loss_funcs[name], (X_train, y_train, X_eval, y_eval), 'FGSM', 5)
        dX, dy, dc = poison_model(model, loss_funcs[name], (X_train, y_train, X_eval, y_eval), 'DeepFool', 5)
        # nX, ny, nl_hist, ni = poison_model(model, loss_funcs[name], (X_train, y_train, X_eval, y_eval), 'Norm', 1)
        # plt.plot([np.linalg.norm(x, 2) for x in fX], linewidth=3, linestyle='solid', marker='o', markersize=9)
        print([np.linalg.norm(x, 2) for x in dX])
        axs1.plot([np.linalg.norm(x, 2) for x in dX], linewidth=3, linestyle='dotted', marker='o', markersize=9)
        # plt.plot([np.linalg.norm(x, 2) for x in nX], linewidth=3, linestyle='dashed', marker='o', markersize=9)

        #plt.ylabel(r'$\vert\vertx\vert\vert$')
        #plt.xlabel(r'k')
        #plt.xticks(ticks=range(len(dX)))

        axs2.set_ylim(0, 1)
        axs2.set_xticks(range(len(dc[0])))
        axs2.plot(dc[0], dc[1], linewidth=3, linestyle='dotted', marker='o', markersize=9, label=name)
        #axs2.plot(dc[1],  linewidth=3, linestyle='dotted', marker='o', markersize=9)


    plt.show()


def epoch_plots(recompute: bool = True):
    epoch_length = 10    # the number of sample points in any epoch
    epoch_num = 3   # the number of epochs that the model will be trained over

    fig1, axs1 = plt.subplots(nrows=2, ncols=1)
    fig2, axs2 = plt.subplots(nrows=2, ncols=1)
    for name, model in models.items():
        # load loss history or compute and save it
        if os.path.exists(f"{model}") and not recompute:
            loss_history = np.loadtxt(f"loss--{model}")
            performance_history = np.loadtxt(f"performance--{model}") # no load atm
        else:
            loss_history, performance_history = train_model(model, X_train, y_train, X_eval, y_eval, loss_funcs[name], epoch_num, epoch_length)
            np.savetxt(f"loss--{name}", loss_history) # uncomment when returns

        axs1[0].plot(loss_history[0], label=name, linewidth=3, linestyle='solid', marker='o', markersize=9)
        axs1[0].set_ylabel('Training', fontsize=15)
        axs1[0].set_xticks([])

        axs1[1].plot(loss_history[1], label=name, linewidth=3, linestyle='solid', marker='o', markersize=9)
        axs1[1].set_ylabel('Test', fontsize=15)
        axs1[1].set_xlabel('Epoch', fontsize=15)
        axs1[1].set_xticks(range(epoch_num), fontsize=10)

        axs2[0].plot(performance_history['true positive'], label=name, linewidth=3, linestyle='solid', marker='o', markersize=9)
        axs2[0].set_ylabel('True positive', fontsize=15)
        axs2[0].set_xticks([])

        axs2[1].plot(performance_history['true negative'], label=name, linewidth=3, linestyle='solid', marker='o', markersize=9)
        axs2[1].set_ylabel('True negative', fontsize=15)
        axs2[1].set_xlabel('Epoch', fontsize=15)
        axs2[1].set_xticks(range(epoch_num), fontsize=10)

        axs1[0].legend()
        axs2[0].legend()

        fig1.suptitle('Binary cross entropy', fontsize=18)
        fig2.suptitle('Performance metrics', fontsize=18)


        #axs2[0].plot(performance_history['true positive'], label=name, linewidth=3, linestyle='solid', marker='o', markersize=9)

    plt.show()


def performance_plot() -> None:
    """
    :return: trains each model using train_model in a batch style and plots the loss over epochs, displays
             the accuracy metrics
    """

    models['DNN'].fit(X_train, y_train)
    models['CNN'].fit(X_train, y_train)

    print('after training metrics')
    dnn_metrics = evaluation_metrics(dnn, X_eval, y_eval)
    cnn_metrics = evaluation_metrics(cnn, X_eval, y_eval)
    print('dnn', dnn_metrics)
    print('cnn', cnn_metrics)

    plt.figure()

    fig, ax = plt.subplots()
    x = np.arange(len(dnn_metrics))
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width / 2, list(dnn_metrics.values()), width, label='DNN')
    rects2 = ax.bar(x + width / 2, list(cnn_metrics.values()), width, label='CNN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=15)
    ax.set_title('Model Performance metrics', fontsize=15)
    ax.set_xticks(x, list(dnn_metrics.keys()), fontsize=15)
    ax.set_ylim(0, 1)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    #plt.figure()
    #m = tf.keras.metrics.AUC(curve='ROC')
    #m.update_state(y_eval, models['DNN'].predict(X_eval))

    #plot_roc_curve(dnn, X_eval, y_eval)

    plt.show()


X_train, X_eval, y_train, y_eval = preprocess_UNSW()


time_steps = 16
units = 32

n = X_train.shape[1]
dnn = CustomNN(n, tf.keras.initializers.he_uniform)
cnn = CNN_Model(n, 2)
lstm = lstmClass()


dnn.compile(optimizer='Adam', loss=tf.losses.binary_crossentropy, metrics=[tf.metrics.TruePositives()])
cnn.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=[tf.metrics.TruePositives()])
lstm.compile()

#lstm.fit(X_train, y_train)

models = {'DNN': dnn, 'CNN': cnn, 'LSTM': lstm}
loss_funcs = {'DNN': log_loss, 'CNN': log_loss, 'LSTM': log_loss}

X_train = X_train.iloc[0:10, :]        # for speed
y_train = y_train.iloc[0:10, :]
X_eval = X_eval.iloc[0:10, :]
y_eval = y_eval.iloc[0:10, :]

#epoch_plots()
#performance_plot()
models['DNN'].fit(X_train, y_train, verbose=0)
models['CNN'].fit(X_train, y_train, verbose=0)
models['LSTM'].fit(X_train, y_train, verbose=0)
poisoning_plot()