import Challenge2.DatasetCreator as dc
import Challenge2.RegressionModels as models

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

splitTrainW_x, splitTrainW_y, splitTestW_x, splitTextW_y = dc.splitByParticipant(trainW_x, trainW_y, testW_x, testW_y)

