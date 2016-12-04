# MLCR-Challange 2

To start the programme please make sure all the following files have been unzipped:

- Main.py
- DatasetCreator.py
- RegressionModels.py
- dataset
  - modifiedLabels.csv
  - data.h5

The main method will call all the methods needed to complete the challenge task. The implementation of these methods can be found in Analyzer.py. You can change the classifier by changeing the string parameter given to the corresponding classification method. We recommend using either Random Forest or KNN as we have tested those the most and achieved better results than with the other two classifiers, even though SVM is also not too far behind.

During the first call of the programme, it will generate a csv file containing the optimized data set that is used for the classification. Initially the value of optimization is set to 80%, which means, that only access points that appear in at least 80% of their own locations will be kept. This makes sure that we don't look at the access points that seem to appear randomly in some datapoints. You can change that value up to 100% which will reduce the total data to roughly 100 access points. Once the cvs file is created, ever consecutive start of the programme will ask if you would like to use the already generated file of if you would like to create a new one. This will make further initiations of the programme much faster. You can change the optimization value in the main.py, however we recommend using the 80% value as it seems that these access points deliver the best classification results. 

The output of the programme will show :

1. The cross fold validation score (accuracy) of the classifiers for 5 iterations
2. The confusion matrix of the results from the test-set evaluation on the unseen data
3. The precision, recall and f1 score of the results from the test-set evaluation
4. The percentage of misclassified classes (for the second classifier)
5. The average error in meter for the misclassified classes 
