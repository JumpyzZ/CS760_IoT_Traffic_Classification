from sklearn.metrics import roc_curve, auc

def Draw_ROC(model1,model2):

    fpr_DNN,tpr_DNN,thresholds=roc_curve(np.array(y_eval),model1.predict(X_eval))
    roc_auc_DNN=auc(fpr_DNN,tpr_DNN)
    fpr_CNN,tpr_CNN,thresholds=roc_curve(np.array(y_eval),model2.predict(X_eval))
    roc_auc_CNN=auc(fpr_CNN,tpr_CNN)


    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))

    plt.plot(fpr_DNN,tpr_DNN,'purple',lw=lw,label='DNN_AUC = %0.2f'% roc_auc_DNN)

    plt.plot(fpr_CNN,tpr_CNN,color='darkorange', lw=lw, label='CNN_AUC = %0.2f'% roc_auc_CNN)

    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)

    plt.show()

Draw_ROC(dnn,cnn)