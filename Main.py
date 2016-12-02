'''
Created on Nov 28, 2016

@author: tobi
'''
import DatasetCreator as dc
import RegressionModels as models

dataset = dc.createDataSet()

trainF_x, trainF_y, testF_x, testF_y = dc.inputOutputDataFreeDays(dataset, False)
trainW_x, trainW_y, testW_x, testW_y = dc.inputOutputDataWorkDays(dataset, False)

splitTrainW_x, splitTrainW_y, splitTestW_x, splitTextW_y = dc.splitByParticipant(trainW_x, trainW_y, testW_x, testW_y)

print (trainW_x)
print (splitTrainW_x[0])
print (splitTrainW_x[5])

#print("Train Set free")
#print (trainF_x)
#print (trainF_y)

#print("--------------------")
#print("Test Set free")
#print (testF_x)
#print (testF_y)
#print("--------------------")
#print("Train set Work")
#print (trainW_x)
#print (trainW_y)
#print("Test set Work")
#print("--------------------")
#print (testW_x)
#print (testW_y)

#do regression
#models.trainCartModel(inputWorkDays, outputWorkDays)
#models.trainAdaBoostModel(inputWorkDays, outputWorkDays)
#models.trainSVR(inputWorkDays, outputWorkDays, kernel = 'rbf')
