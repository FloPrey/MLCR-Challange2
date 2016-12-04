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
print(">>>Training AdaBoost with leave-one-out validation on workday data<<<")
models.adaBoostModelWithCrossFoldValidation(inputDataWork, outputDataWork, "Workdays")

print("\n\n>>>Training AdaBoost with leave-one-out validation on freeday data<<<")
models.adaBoostModelWithCrossFoldValidation(inputDataFree, outputDataFree, "Freedays")

# use already split dataset 
print("\n\n>>>Training AdaBoost with test and train set on workday data<<<")
models.adaBoostModel(trainW_x, trainW_y, testW_x, testW_y, "Workdays")

print("\n\n>>>Training AdaBoost with test and train set on freeday data<<<")
models.adaBoostModel(trainF_x, trainF_y, testF_x, testF_y, "Freedays")

# linear regression

print("\n\n>>>Training Linear Regression Model on workday data<<<")
models.trainLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")

print("\n\n>>>Training Linear Regression Model on freeday data<<<")
models.trainLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")

print("\n\n>>>Training weightened Linear Regression Model on workday data<<<")
models.trainWeightedLinearRegression(trainW_x, trainW_y, testW_x, testW_y, "Workday")

print("\n\n>>>Training weightened Linear Regression Model on freeday data<<<")
models.trainWeightedLinearRegression(trainF_x, trainF_y, testF_x, testF_y, "Freeday")
