import time
import matplotlib.pyplot as plt

from typing import Union, Callable
from sklearn import svm
from secml.adv.attacks import CAttackPoisoningSVM

from preprocess import * 


svc = svm.SVC 


def posion_svm(X_train: pd.DataFrame, y_train: pd.DataFrame, loss: Callable) -> tuple[pd.DataFrame, float]:
    """
        :objective: perform dataset posioning on a SVM 
        :return: posioned dataset 
    """

    return # possibly use secml 



def train_svm(X: pd.DataFrame, y: pd.DataFrame, split: float) -> tuple[list, list, list]:
    """
        :objective: iteratively trains a SVC on dataset and posions the dataset
        :return: model performance over iteration of posioning 
    """

    # meta datas <--- use history class? 
    accuracy = [] 
    eval_time = [] 
    training_time = []
    parameters = []

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=split)  # split dataset 

    for i in range(3):  # I'm what sure not the stopping creteria will be! 
        train_start = time.time()
        svc = svm.SVC(C = 1.0, kernel = "rbf").fit(X_train, y_train) # training svc... 
        
        eval_start = time.time() # begin the clock
        accuracy = svc.score(X_test, y_test) # obtain accurancy result

        end = time.time() # stop clock 

        X_train, y_train = posion_svm(X_train, y_train) # posion attack 

        accuracy.append(svc.score(X_test, y_test))  # assemble meta data  
        eval_time.append(end - eval_start)
        training_time.append(end - train_start)
    

    # <---- maybe a save function for posioned data? 
    

    return accuracy, eval_time, training_time