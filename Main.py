'''
Created on Nov 28, 2016

@author: tobi
'''
import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

dataset = dc.createDataSet()

inputData, outputLabel = dc.createInputAndOutputDataset(dataset)

print(inputData)
print(outputLabel)

#do regression
models.trainCartModel(inputData, outputLabel)
