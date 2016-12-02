'''
Created on Nov 28, 2016

@author: tobi
'''
import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

dataset = dc.createDataSet()

trainF_x, trainF_y, testF_x, testF_y = dc.inputOutputDataFreeDays(dataset, False)
trainW_x, trainW_y, testW_x, testW_y = dc.inputOutputDataWorkDays(dataset, False)

print("Train Set free")
print (trainF_x)
print (trainF_y)

print("--------------------")
print("Test Set free")
print (testF_x)
print (testF_y)
print("--------------------")
print("Train set Work")
print (trainW_x)
print (trainW_y)
print("Test set Work")
print("--------------------")
print (testW_x)
print (testW_y)

#do regression
#models.trainCartModel(inputWorkDays, outputWorkDays)
#models.trainAdaBoostModel(inputWorkDays, outputWorkDays)
#models.trainSVR(inputWorkDays, outputWorkDays, kernel = 'rbf')
