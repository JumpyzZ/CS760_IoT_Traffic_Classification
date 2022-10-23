import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_UNSW
from sklearn.metrics import roc_curve, auc

def Draw_ROC(model1,model2,model3, model4, model5, model6):

    X_train, X_eval, y_train, y_eval = preprocess_UNSW()

    fpr_DNN,tpr_DNN,thresholds=roc_curve(np.array(y_eval),model1.predict(X_eval))
    roc_auc_DNN=auc(fpr_DNN,tpr_DNN)

    fpr_CNN,tpr_CNN,thresholds=roc_curve(np.array(y_eval),model2.predict(X_eval))
    roc_auc_CNN=auc(fpr_CNN,tpr_CNN)

    fpr_LSTM,tpr_LSTM,thresholds=roc_curve(np.array(y_eval),model3.predict(np.array(X_eval)[:,:, np.newaxis]))
    roc_auc_LSTM=auc(fpr_LSTM,tpr_LSTM)

    fpr_RNN,tpr_RNN,thresholds=roc_curve(np.array(y_eval),model4.predict(np.array(X_eval).reshape([-1, X_eval.shape[1], 1])))
    roc_auc_RNN=auc(fpr_RNN,tpr_RNN)

    fpr_RF,tpr_RF,thresholds=roc_curve(np.array(y_eval),model5.predict(np.array(X_eval)))
    roc_auc_RF=auc(fpr_RF,tpr_RF)

    fpr_SVM,tpr_SVM,thresholds=roc_curve(np.array(y_eval),model6.predict(np.array(X_eval)))
    roc_auc_SVM=auc(fpr_SVM,tpr_SVM)


    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))

    plt.plot(fpr_DNN,tpr_DNN,'purple',lw=lw,label='DNN_AUC = %0.2f'% roc_auc_DNN)

    plt.plot(fpr_CNN,tpr_CNN,color='darkorange', lw=lw, label='CNN_AUC = %0.2f'% roc_auc_CNN)

    plt.plot(fpr_LSTM,tpr_LSTM,color='red', lw=lw, label='LSTM_AUC = %0.2f'% roc_auc_LSTM)

    plt.plot(fpr_RNN,tpr_RNN,color='green', lw=lw, label='RNN_AUC = %0.2f'% roc_auc_RNN)

    plt.plot(fpr_RF,tpr_RF,color='black', lw=lw, label='RF_AUC = %0.2f'% roc_auc_RF)

    plt.plot(fpr_SVM,tpr_SVM,color='brown', lw=lw, label='SVM_AUC = %0.2f'% roc_auc_SVM)

    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.title('ROC Curve')
    plt.savefig('./ROCcurve.eps', dpi=300)
    plt.show()

Draw_ROC(dnn, cnn, lstm.model, rnn, rf, svc)
