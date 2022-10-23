import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_UNSW, get_UNSW
import pandas as pd


def poisoning_plots(data: tuple, names: list):

    X_train, X_eval, y_train, y_eval = data

    k1, k2 = 0, 1
    for name in names:

        dX = np.load(f'DeepFool_{name}_X.npy')
        dy = np.load(f'DeepFool_{name}_y.npy')
        dc = np.load(f'DeepFool_{name}_c.npy')
        dm = np.load(f'DeepFool_{name}_m.npy')
        dl = np.load(f'DeepFool_{name}_l.npy')
        di = np.load(f'DeepFool_{name}_i.npy')

        fX = np.load(f'FGSM_{name}_X.npy')
        fy = np.load(f'FGSM_{name}_y.npy')
        fc = np.load(f'FGSM_{name}_c.npy')
        fm = np.load(f'FGSM_{name}_m.npy')
        fl = np.load(f'FGSM_{name}_l.npy')
        fi = np.load(f'FGSM_{name}_i.npy')

        print(fX)
        print(dX)

        fig, axs = plt.subplots(nrows=1, ncols=1)

        step = 50

        axs.scatter(X_train.values[0:-1:step][:, 0], X_train.values[0:-1:step][:, 1], marker="o", c=y_train.values[0:-1:step], s=25, edgecolor="k")
        axs.set_xlabel(r'$x_1$', fontsize=15)
        axs.set_ylabel(r'$x_2$', fontsize=15)

        fig.suptitle(f'{name}', fontsize=20)

        for n, y in enumerate(dy):
            if y == 1:
                axs.scatter(dX[:, k1], dX[:, k2], marker="o", c='red', s=25, edgecolor="k", zorder=100)
            else:
                axs.scatter(dX[:, k1], dX[:, k2], marker="s", c='red', s=25, edgecolor="k", zorder=100)
            if fy[n] == 1:
                axs.scatter(fX[:, k1], fX[:, k2], marker="o", c='green', s=25, edgecolor="k", zorder=100)
            else:
                axs.scatter(fX[:, k1], fX[:, k2], marker="s", c='green', s=25, edgecolor="k", zorder=100)

    plt.show()


X_train = pd.read_csv('xtrain.csv').iloc[:, 1:]
X_eval = pd.read_csv('xeval.csv').iloc[:, 1:]
y_train = pd.read_csv('ytrain.csv').iloc[:, 1:]
y_eval = pd.read_csv('yeval.csv').iloc[:, 1:]

#print(X_train.shape, X_eval.shape, y_train.shape, y_eval.shape)

#data = (X_train, X_eval, y_train, y_eval)

poisoning_plots((X_train, X_eval, y_train, y_eval), ['RNN'])