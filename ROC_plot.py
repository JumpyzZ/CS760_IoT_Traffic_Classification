from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def Draw_ROC(models: dict, data: tuple):
    X_train, y_train, X_eval, y_eval = data

    plt.figure(figsize=(10, 10))
    for name, model in models.items():
        fpr, tpr, thresholds = roc_curve(np.array(y_eval), model.predict(X_eval))
        roc_auc_DNN = auc(fpr, tpr)

        plt.plot(fpr, tpr, 'purple', lw=2, label=f'{name} AUC = %0.2f' % roc_auc_DNN)

    plt.legend(loc='lower right', fontsize=12)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)

    plt.show()


