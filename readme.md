# MLCR-Challange 2

To start the programme please make sure all the following files have been unzipped:

- Main.py
- DatasetCreator.py
- RegressionModels.py
- dataset
  - modifiedLabels.csv
  - data.h5

The main method will call all the methods needed to complete the challenge task. The implementation of the Regression Models can be found in RegressionModels.py. The DatasetCreator.py class creates the data sets from the data.h5 and the modifiedLabels.csv file. The modifiedLabels.csv file contains the diary information of the participants with an idditional column "day" and some data correction as some participants made some mistakes filling out the daily sheets.

The main method will invoke the following models:

1. AdaBoost Regression Model using leave-one-out prediction / validation on Workday data
2. AdaBoost Regression Model using leave-one-out prediction / validation on Freeday data
3. AdaBoost Regression Model using the predefined test and train set on Workday data
4. AdaBoost Regression Model using the predefined test and train set on Freeday data
5. Linear Regression Model using the predefined test and train set on Workday data
6. Linear Regression Model using the predefined test and train set on Freeday data
7. Weighted Linear Regression Model using the predefined test and train set on Workday data
8. WeightedLinear Regression Model using the predefined test and train set on Freeday data


The output of the programme will show :

1. R² score of the regression models
  - with and without MSFsc feature
2. RSMe score of the regression models
  - with and without MSFsc feature
3. Prediction to ground truth diagram of the regression models
  - with and without MSFsc feature

--> Please make sure to close or save the diagrams to get the rest of the results to show.

Nice to know:

If you wish to print the 300 decision trees used in the ada Boost "forest", then you can add uncomment line 30 in the RegressionModels.py. This will create 300 files containing each of the decision tree in the location decisionTrees/ . The files need to be converted to png with the following bash command: for i in *;do dot -Tpng "$i" -o "$i"..png;done.

