'''
Created on Nov 28, 2016

@author: tobi
'''
import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

dataset = dc.createDataSet()

print(dataset)

#inputData, outputData = dc.createInputAndOutputDataset(dataset)

#do regression
#models.trainModel(inputData, outputData)
