'''
Created on Nov 26, 2016

@author: tobi
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor as AdaboostR
def trainCartModel(inputData, labels):
    
    regressor = DecisionTreeRegressor(random_state=0)
    score = cross_val_score(regressor, inputData, labels, cv=2)
    
    print("CART - Regression Model trained with score: ")
    print (score)

def AdaBoostTryOuts(inputData, labels):
    regressor = AdaboostR(base_estimator=DecisionTreeRegressor(random_state=0), n_estimators=300)
    score = cross_val_score(regressor, inputData, labels, cv=2)
    print("ADABoost - Regression Model trained with score: ")
    print (score)

