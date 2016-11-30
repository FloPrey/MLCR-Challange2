'''
Created on Nov 26, 2016

@author: tobi
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


def trainCartModel(inputData, labels):
    
    regressor = DecisionTreeRegressor(random_state=0)
    score = cross_val_score(regressor, inputData, labels, cv=2)
    
    print("CART - Regression Model trained with score: ")
    print (score)
    

def errorCalculation(prediction, groundTruth):

    rmse = sqrt(mean_squared_error(groundTruth, prediction))
