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
print("Workdays AdaBoost with leave-one-out validation:")
models.adaBoostModelWithCrossFoldValidation(inputDataWork, outputDataWork, "Workdays")

print("\nFreedays AdaBoost with leave-one-out validation:")
models.adaBoostModelWithCrossFoldValidation(inputDataFree, outputDataFree, "Freedays")

# use already split dataset 
print("\nWorkdays AdaBoost on test and train set:")
models.adaBoostModel(trainW_x, trainW_y, testW_x, testW_y, "Workdays")

print("\nFreedays AdaBoost on test and train set:")
models.adaBoostModel(trainF_x, trainF_y, testF_x, testF_y, "Freedays")

# linear regression

print("\n\nWorkdays Linear:")
models.trainLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")

print("\n\nWorkdays weightened:")
models.trainWeightedLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")

print("\n\nFreedays Linear:")
models.trainLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")

print("\n\nFreedays Weightened:")
models.trainWeightedLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")
