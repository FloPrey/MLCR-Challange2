import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

dataset = dc.createDataSet()

datasetFree = dc.inputOutputDataFreeDays(dataset)
datasetWork = dc.inputOutputDataWorkDays(dataset)

# split dataset into input (x-value) and output (y-value)
inputDataFree, outputDataFree = dc.splitLabels(datasetFree)
inputDataWork, outputDataWork = dc.splitLabels(datasetWork)

# split entire dataset into train and test sets and their corresponding outputl values
trainF_x, trainF_y, testF_x, testF_y = dc.splitDataset(datasetFree)
trainW_x, trainW_y, testW_x, testW_y = dc.splitDataset(datasetWork)

splitTrainW_x, splitTrainW_y, splitTestW_x, splitTextW_y = dc.splitByParticipant(trainW_x, trainW_y, testW_x, testW_y)
print (trainW_x)
print (splitTrainW_x[0])
print (splitTrainW_x[5])

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

# use entire dataset for training with cross-fold-validation
#models.adaBoostModel(inputDataWork, outputDataWork)

# use seperate train and test set
#models.trainSVR(inputWorkDays, outputWorkDays, kernel = 'rbf')
