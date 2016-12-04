from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import AdaBoostRegressor as AdaboostR

def trainDecisionTreeModel(inputData, outputData, workOrFreeDay):
    
    regressor = DTR(random_state=0, max_depth = 5)
    predicted = cross_val_predict(regressor, inputData, outputData, cv=10)
    printEvaluationScores(predicted, outputData, "Simple DecisionTree", workOrFreeDay)
    
def adaBoostModel(train_x, train_y, test_x, test_y, workOrFreeDay):
    
    rng = np.random.RandomState(1)
    
    adaBoost = AdaboostR(DTR(max_depth=5), n_estimators=300, random_state=rng)
    adaBoost.fit(train_x, train_y)
    predicted = adaBoost.predict(test_x)
    
    # show test results 
    printEvaluationScores(predicted, test_y, "AdaBoost model with MSFsc", workOrFreeDay)  
    
    # invokes method to print the tree structure of the 300 trained tree
    #saveTreeStrucutre(adaBoost)
    
    # Predict without MSFSC
    x_trainWithoutMSFSC = train_x.copy()
    x_testWithoutMSFSC = test_x.copy()
    
    del x_trainWithoutMSFSC['MSFSC']
    del x_testWithoutMSFSC['MSFSC']
    
    adaBoost.fit(x_trainWithoutMSFSC, train_y)
    predicted = adaBoost.predict(x_testWithoutMSFSC)
    
    # show test results 
    printEvaluationScores(predicted, test_y, "AdaBoost model with without MSFsc", workOrFreeDay) 

def adaBoostModelWithCrossFoldValidation(inputData, outputData, workOrFreeDay):
      
    rng = np.random.RandomState(1)
    adaBoost = AdaboostR(DTR(max_depth=5), n_estimators=300, random_state=rng)
    
    # do leave one-out cross prediction
    adaBoostPredict = cross_val_predict(adaBoost, inputData, outputData, cv=len(inputData))   
    
    # show test results 
    printEvaluationScores(adaBoostPredict, outputData, "AdaBoost model with MSFsc", workOrFreeDay)  
    
        # Predict without MSFSC
    dataWithoutMSFSC = inputData.copy()
    
    del dataWithoutMSFSC['MSFSC']
    adaBoostPredict = cross_val_predict(adaBoost, dataWithoutMSFSC, outputData, cv=len(inputData))
    
    # show test results 
    printEvaluationScores(adaBoostPredict, outputData, "AdaBoost model without MSFsc", workOrFreeDay)
    
    
def printEvaluationScores(predicted, groundTruth, modelName, workOrFreeDay):
    
    r2Value = r2_score(groundTruth, predicted)
    RMSe = sqrt(mean_squared_error(groundTruth, predicted)) 
    
    print("\n----------------------------")
    print("%s - trained" %modelName) 
    print("----on %s data:" %workOrFreeDay + "-------")
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

def trainWeightedLinearRegression(X_train, y_train, X_test, y_test, workOrFreeDay):

    reg = linear_model.Ridge(alpha=0.5)

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)

    # show test results
    printEvaluationScores(predict, y_test, "Weightened Linear Regression with MSFsc", workOrFreeDay)
    
    # Predict without MSFSC
    x_trainWithoutMSFSC = X_train.copy()
    x_testWithoutMSFSC = X_test.copy()
    
    del x_trainWithoutMSFSC['MSFSC']
    del x_testWithoutMSFSC['MSFSC']

    reg = linear_model.Ridge(alpha=0.5)

    reg.fit(x_trainWithoutMSFSC, y_train)
    predict = reg.predict(x_testWithoutMSFSC)

    # show test results
    printEvaluationScores(predict, y_test, "Weightened Linear Regression without MSFsc", workOrFreeDay)

def trainLinearRegression(X_train, y_train, X_test, y_test, workOrFreeDay):

    reg = linear_model.LinearRegression()

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)

    # show test results
    printEvaluationScores(predict, y_test, "Linear Regression with MSFsc", workOrFreeDay)

    # Predict without MSFSC
    x_trainWithoutMSFSC = X_train.copy()
    x_testWithoutMSFSC = X_test.copy()
    
    del x_trainWithoutMSFSC['MSFSC']
    del x_testWithoutMSFSC['MSFSC']

    reg.fit(x_trainWithoutMSFSC, y_train)
    predict = reg.predict(x_testWithoutMSFSC)

    # show test results
    printEvaluationScores(predict, y_test, "Linear Regression without MSFsc", workOrFreeDay)

def prettyprint(input):
    output = ""
    if isinstance(input, list ):
        for item in input:
            output = output + '\t' + str(item).replace('.', ',')
    else:
        output = str(input).replace('.', ',')
    return output
