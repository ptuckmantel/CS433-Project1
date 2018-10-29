FILES IN ZIP: 

OBLIGATORY FOR SUBMISSION 
run.py
- trains model on full train.csv data set
- parameters for training are in params dictionary at the beginning of the file
- NO INPUT PARAMETERS ARE NEEDED
- stores final weights in "weights.json" file 
- prints performance statistics for trainin test 
- loads test data and predicts labels 
- stores labels in "PredictionLabels.csv"

run_testSplited.py
- on some computers there is not enough memory to process test file using preprocessing and number of additional features we create
- from this reason, test is splited in 2 segments and predicted for each separately
- in the end they are concatinated before exported to .csv file 
- other  than this everything is the same as run.py

proj1_helpers.py
- function "create_csv_submission" is not working properly for Windows 
- every second line is empty and then submission is 2x longer and gets bad results 
- in open(name, 'w', newline='')  newline='' was added so that this is fixed for Windows os 

implementations.py
- function requested to be implemented 

DATA FOR TRAINING AND TESTING 
- it is necessary to put "data" folder with data for training and testing in the same folder as this scripts 

ADITIONAL LIBRARIES (NECESSARY TO RUN CODE)
- in order for run.py to run and be as short as possible lot of functions was written and separated in 4 libraries
1. lib_dataPreprocessing.py 
- contains all functions for data preprocessing and feature engineering
2. lib_MLmodels.py
- contains all functions relate to ML models
- also all of the functions from implementations.py
3. lib_errorMetrics.py
- contains different functions for measuring output performance and statistics
4. lib_PCA.py
- uses sklearn function that was used for testing whether PCA can help in reducing dimensions of features
- it didn't help in the end so it's not used in final code 
- if it helped we would implement our own function to replace sklearn PCA function 

ADITIONAL SCRIPTS
script_findingOptimalParameters.py
- example of script where we tested different setups (parameters) in order to find best ones 
- here feature engineering is tested, in different one optimal gamma and lambda were tested 
- performance metrics is stored in output .json file for latter observing 
DataExploration folder 
-jupyter notebook file with various data exploration ideas we tried with lots of visualizations

 
