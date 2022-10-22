from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import roc_curve, auc


def Draw_ROC(model1, model2, model3, model4, data):

    X_train, y_train, X_eval, y_eval = data

    fpr_DNN,tpr_DNN,thresholds=roc_curve(np.array(y_eval),model1.predict(X_eval))
    roc_auc_DNN=auc(fpr_DNN,tpr_DNN)

    fpr_CNN,tpr_CNN,thresholds=roc_curve(np.array(y_eval),model2.predict(X_eval))
    roc_auc_CNN=auc(fpr_CNN,tpr_CNN)

    fpr_LSTM,tpr_LSTM,thresholds=roc_curve(np.array(y_eval),model3.predict(np.array(X_eval)[:,:, np.newaxis]))
    roc_auc_LSTM=auc(fpr_LSTM,tpr_LSTM)

    fpr_RNN,tpr_RNN,thresholds=roc_curve(np.array(y_eval),model4.predict(np.array(X_eval).reshape([-1, X_eval.shape[1], 1])))
    roc_auc_RNN=auc(fpr_RNN,tpr_RNN)


    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))

    plt.plot(fpr_DNN,tpr_DNN,'purple',lw=lw,label='DNN_AUC = %0.2f'% roc_auc_DNN)

    plt.plot(fpr_CNN,tpr_CNN,color='darkorange', lw=lw, label='CNN_AUC = %0.2f'% roc_auc_CNN)

    plt.plot(fpr_LSTM,tpr_LSTM,color='red', lw=lw, label='LSTM_AUC = %0.2f'% roc_auc_LSTM)

    plt.plot(fpr_RNN,tpr_RNN,color='green', lw=lw, label='RNN_AUC = %0.2f'% roc_auc_RNN)

    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)

    plt.show()