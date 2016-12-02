'''
Created on Nov 26, 2016

@author: tobi
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor as AdaboostR
from sklearn.metrics import mean_squared_error
from math import sqrt


def trainCartModel(inputData, labels):
    
    regressor = DecisionTreeRegressor(random_state=0)
    score = cross_val_score(regressor, inputData, labels, cv=2, scoring="neg_mean_squared_error")
    
    print("CART - Regression Model trained with score: ")
    print (score)


def trainAdaBoost(inputData, labels):
    regressor = AdaboostR(base_estimator=DecisionTreeRegressor(random_state=0), n_estimators=300)
    score = cross_val_score(regressor, inputData, labels, cv=2, scoring="neg_mean_squared_error")
    print("ADABoost - Regression Model trained with score: ")
    score = cross_val_score(regressor, inputData, labels, cv=2, scoring="r2")
    print("ADABoost R2")
    print (score)
    print("ADABoost - Regression Model trained with predict: ")
    score = cross_val_predict(regressor, inputData, labels, cv=2)
    print(score)


def errorCalculation(prediction, groundTruth):
    rmse = sqrt(mean_squared_error(groundTruth, prediction))

#kernel = linear, rbf, polynomial(degree needed)
def trainSVR(features, target, kernel, C=1e3, degree=None):

    if (degree == None):
        svr_mdl = SVR(kernel=kernel, C=C)

    else:
        svr_mdl = SVR(kernel=kernel, C=C, degree=degree)

    svr_mdl.fit(features, target)
    score = cross_val_score(svr_mdl, features, target, cv=2)
    print("SVR ", kernel, " - trained with score: ")
    print (score)
