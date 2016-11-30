'''
Created on Nov 28, 2016

@author: tobi
'''
import DatasetCreator as dc
import RegressionModels as models

dataset = dc.createDataSet()

inputData, outputLabel = dc.createInputAndOutputDataset(dataset, False)

print(inputData)
print("--")
print(outputLabel)

#do regression
models.trainCartModel(inputData, outputLabel)
models.AdaBoostTryOuts(inputData, outputLabel)
