# MLCR-Challange 2

To start the programme please make sure all the following files have been unzipped:

- Main.py
- DatasetCreator.py
- RegressionModels.py
- dataset
  - modifiedLabels.csv
  - data.h5

The main method will call all the methods needed to complete the challenge task. The implementation of the Regression Models can be found in RegressionModels.py. The DatasetCreator.py class creates the data sets from the data.h5 and the modifiedLabels.csv file. The modifiedLabels.csv file contains the diary information of the participants with an idditional column "day" and some data correction as some participants made some mistakes filling out the daily sheets.

The output of the programme will show :

1. R² score of the regression models
2. RSMe score of the regression models
3. Prediction to ground truth diagram of the regression models.
