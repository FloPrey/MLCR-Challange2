import DatasetCreator as dc
import RegressionModels as models

dataset = dc.createDataSet()

datasetFree = dc.inputOutputDataFreeDays(dataset)
datasetWork = dc.inputOutputDataWorkDays(dataset)

# split dataset into input (x-value) and output (y-value)
inputDataFree, outputDataFree = dc.splitLabels(datasetFree)
inputDataWork, outputDataWork = dc.splitLabels(datasetWork)

# split entire dataset into train and test sets and their corresponding output values
# horizontall split for every participant
trainF_x, trainF_y, testF_x, testF_y = dc.splitDataset(datasetFree)
trainW_x, trainW_y, testW_x, testW_y = dc.splitDataset(datasetWork)

# use entire dataset for training with cross-fold-validation
models.adaBoostModelWithCrossFoldValidation(inputDataWork, outputDataWork, "Workdays")
models.adaBoostModelWithCrossFoldValidation(inputDataFree, outputDataFree, "Freedays")

# use already split dataset 
models.adaBoostModel(trainW_x, trainW_y, testW_x, testW_y, "Workdays")
models.adaBoostModel(trainF_x, trainF_y, testF_x, testF_y, "Freedays")



# linear regression
splitTrainW_x, splitTrainW_y, splitTestW_x, splitTextW_y = dc.splitByParticipant(trainW_x, trainW_y, testW_x, testW_y)
splitTrainF_x, splitTrainF_y, splitTestF_x, splitTextF_y = dc.splitByParticipant(trainF_x, trainF_y, testF_x, testF_y)

for i in range (len(splitTrainW_x)):
    print ("Linear Regression Participant, ", i)
    models.trainLinearRegression(trainW_x[i], trainW_y[i], testW_x[i], testW_y[i])
    models.trainWeightedLinearRegression(trainW_x[i], trainW_y[i], testW_x[i], testW_y[i])