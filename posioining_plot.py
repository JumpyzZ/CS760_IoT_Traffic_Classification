import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_UNSW


def poisoning_plots(data: tuple):

    X_train, X_eval, y_train, y_eval = data

    fig, axs = plt.subplots()

    k1, k2 = 0, 1

    dX = np.load(f'DeepFool_DNN_X.npy')
    dy = np.load(f'DeepFool_DNN_y.npy')
    dc = np.load(f'DeepFool_DNN_c.npy')
    dm = np.load(f'DeepFool_DNN_m.npy')
    dl = np.load(f'DeepFool_DNN_l.npy')
    di = np.load(f'DeepFool_DNN_i.npy')

    print(X_train.iloc[:, k1].values)
    print(X_train.iloc[:, k2].values)
    print(y_train)

    plt.scatter(X_train.values[:, k1], X_train.values[:, k2], marker="o", c=y_train.values, s=25, edgecolor="k")

    plt.scatter(dX[:, k1], dX[:, k2], marker="o", c=dy.values, s=25, edgecolor="k")

    plt.show()


data = preprocess_UNSW()
X_train, X_eval, y_train, y_eval = data 4
poisoning_plots(data)