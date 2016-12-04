'''
Created on Nov 26, 2016

@author: tobi
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor as AdaboostR
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt

def trainCartModel(inputData, labels):
    
    regressor = DecisionTreeRegressor(random_state=0)
    score = cross_val_score(regressor, inputData, labels, cv=2, scoring="neg_mean_squared_error")

    regressor = DecisionTreeRegressor(random_state=0)
    score = cross_val_score(regressor, inputData, labels, cv=2)

    print("CART - Regression Model trained with score: ")
    print (score)


def trainWeightedLinearRegression(X_train, y_train, X_test, y_test):

    del X_train['Participant_ID']
    del X_test['Participant_ID']
    del X_train['day']
    del X_test['day']

    reg = linear_model.Ridge(alpha=0.5)

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)

    print("------------Test Results Weighted Linear Regression:------------------")
    print("True Values:")
    print(y_test)
    print("Predicted Values:")
    print(predict)
    print("Reached Score(R²):")
    print(score)
    print("-------------End Result--------------------")


def trainLinearRegression(X_train, y_train, X_test, y_test):


    reg = linear_model.LinearRegression()

    model = reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)

    print("------------Test Result Linear Regression:------------------")
    print("True Values:")
    print(y_test)
    print("Predicted Values:")
    print(predict)
    print("Reached Score(R²):")
    print(score)
    print("-------------End Result--------------------")


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
