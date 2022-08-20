import time
import matplotlib.pyplot as plt

from typing import Union, Callable
from sklearn import svm
from secml.adv.attacks import CAttackPoisoningSVM

from preprocess import * 


svc = svm.SVC 


# perform dataset posioning on a SVM 
def posion_svm(X_train: pd.DataFrame, y_train: pd.DataFrame, loss: Callable) -> tuple[pd.DataFrame, float]:

    train_features = X_train
    train_label = y_train

    x_train, x_val, y_train, y_val = train_test_split(train_features, train_label, train_size = 0.8, random_state = 100)

    # set bounds of the attack space.
    lb, ub = x_val.X.min(), x_val.X.max()

    tr = x_train + y_train
    val = x_val + y_val

    # select and initialize parameter
    solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
    }

    pois_attack = CAttackPoisoningSVM(classifier = ss,
                       training_data = tr,
                       val = val ,
                       lb=lb, ub=ub,
                       solver_params=solver_params,
                       random_seed=760)

    # chose and set the initial poisoning sample features and label
    xc = tr[,:].X
    yc = tr[,:].Y
    pois_attack.x0 = xc
    pois_attack.xc = xc
    pois_attack.yc = yc

    print("Initial poisoning sample features: {:}".format(xc.ravel()))
    print("Initial poisoning sample label: {:}".format(yc.item()))

    # Run the poisoning attack
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(val.X, val.Y)

    # Evaluate the accuracy of the original classifier
    acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
    # Evaluate the accuracy after the poisoning attack
    pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)

    print("Original accuracy on test set: {:.2%}".format(acc))
    print("Accuracy after attack on test set: {:.2%}".format(pois_acc))

    return tr,val
  

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

    for i in range(10):  # I'm what sure not the stopping criteria will be!
        train_start = time.time()   # begin the clock
        svc = svm.SVC(C=1.0, kernel="rbf").fit(X_train, y_train) # training svc...
        
        eval_start = time.time()
        accuracy = svc.score(X_test, y_test) # obtain accuracy result

        end = time.time() # stop clock 

        X_train, y_train = posion_svm(X_train, y_train) # poison attack

        accuracy.append(svc.score(X_test, y_test))  # assemble meta data  
        eval_time.append(end - eval_start)
        training_time.append(end - train_start)

    # <---- maybe a save function for posioned data? 

    return accuracy, eval_time, training_time
