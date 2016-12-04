from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from math import sqrt
import types
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import AdaBoostRegressor as AdaboostR

def trainDecisionTreeModel(inputData, outputData, workOrFreeDay):
    
    del inputData['Participant_ID']
    del inputData['day']
    
    regressor = DTR(random_state=0, max_depth = 5)
    predicted = cross_val_predict(regressor, inputData, outputData, cv=10)
    printEvaluationScores(predicted, outputData, "Simple DecisionTree", workOrFreeDay)
    
def adaBoostModel(train_x, train_y, test_x, test_y, workOrFreeDay):
    
    del train_x['Participant_ID']
    del test_x['Participant_ID']
    del train_x['day']
    del test_x['day']
    
    rng = np.random.RandomState(1)
    
    adaBoost = AdaboostR(DTR(max_depth=5), n_estimators=300, random_state=rng)
    adaBoost.fit(train_x, train_y)
    predicted = adaBoost.predict(test_x)
    
    # show test results 
    printEvaluationScores(predicted, test_y, "AdaBoost with train/test set", workOrFreeDay)  
    
    # invokes method to print the tree structure of the 300 trained tree
    #saveTreeStrucutre(adaBoost)

def adaBoostModelWithCrossFoldValidation(inputData, outputData, workOrFreeDay):
    
    del inputData['Participant_ID']
    del inputData['day']
      
    rng = np.random.RandomState(1)
    adaBoost = AdaboostR(DTR(max_depth=5), n_estimators=300, random_state=rng)
    
    # do leave one-out cross prediction
    adaBoostPredict = cross_val_predict(adaBoost, inputData, outputData, cv=len(inputData))   
    
    # show test results 
    printEvaluationScores(adaBoostPredict, outputData, "AdaBoost using leave one out predict", workOrFreeDay)    
    
def printEvaluationScores(predicted, groundTruth, modelName, workOrFreeDay):
    
    r2Value = r2_score(groundTruth, predicted)
    RMSe = sqrt(mean_squared_error(groundTruth, predicted)) 
    
    print("--------------------------")
    print("%(1)s - Regression Model trained on %(2)s data:" % {"1" : modelName, "2" : workOrFreeDay})
    print("R² score of:\t", prettyprint(r2Value))
    print("And RMSe of:\t", prettyprint(RMSe))
    
    # create prediction to ground truth diagram
    fig, ax = plt.subplots()
    ax.scatter(groundTruth, predicted)
    ax.plot([min(groundTruth, key=float), max(groundTruth, key=float)], [min(groundTruth, key=float), max(groundTruth, key=float)], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title(modelName + " on " + workOrFreeDay)
    plt.show()

"""Method to save the 300 decision trees in the decisionTrees folder. To change the files into png use the bash script in the readme."""    
def saveTreeStrucutre(adaBoostModel):

    i_tree = 0
    for tree_in_forest in adaBoostModel.estimators_:
        my_file = 'decisionTrees/tree_' + str(i_tree) + '.dot' 
        tree.export_graphviz(tree_in_forest, out_file = my_file)
        i_tree = i_tree + 1  

def trainWeightedLinearRegression(X_train, y_train, X_test, y_test):

    del X_train['Participant_ID']
    del X_test['Participant_ID']
    del X_train['day']
    del X_test['day']

    reg = linear_model.Ridge(alpha=0.5)

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    RMSe = sqrt(mean_squared_error(y_test, predict))

    print("------------Test Results Weighted Linear Regression:------------------")
    print("------------------------------With MSFsc------------------------------")
    print("True Values:\t", prettyprint(y_test))
    print("Predicted Values:\t", prettyprint(predict))
    print("R²:\t", prettyprint(score))
    print("RMSe:\t", prettyprint(RMSe))
    print("-------------End Result--------------------")

    # Predict without MSFSC
    del X_train['MSFSC']
    del X_test['MSFSC']

    reg = linear_model.Ridge(alpha=0.5)

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    RMSe = sqrt(mean_squared_error(y_test, predict))

    print("------------Test Results Weighted Linear Regression:------------------")
    print("----------------------------Without MSFsc-----------------------------")
    print("True Values:\t", prettyprint(y_test))
    print("Predicted Values:\t", prettyprint(predict))
    print("R²:\t", prettyprint(score))
    print("RMSe:\t", prettyprint(RMSe))
    print("-------------End Result--------------------")



def trainLinearRegression(X_train, y_train, X_test, y_test):

    del X_train['Participant_ID']
    del X_test['Participant_ID']
    del X_train['day']
    del X_test['day']

    reg = linear_model.LinearRegression()

    model = reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    RMSe = sqrt(mean_squared_error(y_test, predict))

    print("--------------------Test Result Linear Regression:--------------------")
    print("------------------------------With MSFsc------------------------------")
    print("True Values:\t", prettyprint(y_test))
    print("Predicted Values:\t", prettyprint(predict))
    print("R²:\t", prettyprint(score))
    print("RMSe:\t", prettyprint(RMSe))
    print("-------------End Result--------------------")

    # Predict without MSFSC
    del X_train['MSFSC']
    del X_test['MSFSC']

    model = reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    RMSe = sqrt(mean_squared_error(y_test, predict))

    print("-------------------Test Result Linear Regression:---------------------")
    print("----------------------------Without MSFsc-----------------------------")
    print("True Values:\t", prettyprint(y_test))
    print("Predicted Values:\t", prettyprint(predict))
    print("R²:\t", prettyprint(score))
    print("RMSe:\t", prettyprint(RMSe))
    print("-------------End Result--------------------")

def prettyprint(input):
    output = ""
    if isinstance(input, list ):
        for item in input:
            output = output + '\t' + str(item).replace('.', ',')
    else:
        output = str(input).replace('.', ',')
    return output
