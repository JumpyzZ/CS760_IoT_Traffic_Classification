import numpy as np
import sklearn
import sklearn.ensemble as ens
from preprocess import *

def randomForrest():

    X_train, X_test, y_train, y_test = preprocess_UNSW()

    # Random Grid search to quickly find the best parameters for the Random Forrest
    random_grid = {'n_estimators': np.linspace(100, 2500, int((2500-100)/200) + 1, dtype=int),
                   'max_depth': [1, 5, 10, 20, 25, 50, 75, 100, 150, 200],
                   'min_samples_split': [1, 2, 5, 10, 15, 20, 25, 30],
                   'min_samples_leaf': [1, 2, 3],
                   'bootstrap': [True, False],
                   'criterion': ['gini', 'entropy']}
    print("The grid")
    random_class = ens.RandomForestClassifier(n_estimators=1000)
    random_grid_class = sklearn.model_selection.RandomizedSearchCV(estimator=random_class, param_distributions= random_grid,
                                                                   n_iter=25, cv= 5, random_state=42, verbose=2)

    print("Fitting now")
    random_class.fit(X_train, np.ravel(y_train))
    print("Predicting now")
    predictions = random_grid_class.predict(X_test)

    print("The training accuracy is : ", random_grid_class.score(X_train, y_train))
    print("The testing accuracy is : ", random_grid_class.score((X_test, y_test)))

    print("The confusion matrix is : ", sklearn.metrics.confusion_matrix(y_test, predictions))


randomForrest()


