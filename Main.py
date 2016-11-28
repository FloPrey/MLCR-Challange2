'''
Created on Nov 28, 2016

@author: tobi
'''
import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

inputData = dc.createInputData()

outputData = dc.createOutputData()

print (inputData)

#do regression
#models.trainModel(inputData, outputData)
