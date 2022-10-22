import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NN_model import *
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, classes = ['benign traffic','malicious traffic'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(np.array(y_eval).reshape(len(y_eval)).astype(int), myround(model.predict(X_eval)))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')