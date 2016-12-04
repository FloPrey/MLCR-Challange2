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


print("\n\nWorkdays Linear:")
trainW_x, trainW_y, testW_x, testW_y = dc.splitDataset(datasetWork)
models.trainLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")


print("\n\nWorkdays weightened:")
trainW_x, trainW_y, testW_x, testW_y = dc.splitDataset(datasetWork)
models.trainWeightedLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")


print("\n\nFreedays Linear:")
trainF_x, trainF_y, testF_x, testF_y = dc.splitDataset(datasetFree)
models.trainLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")


print("\n\nFreedays Weightened:")
trainF_x, trainF_y, testF_x, testF_y = dc.splitDataset(datasetFree)
models.trainWeightedLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")