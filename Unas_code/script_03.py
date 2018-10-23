# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:25:43 2018

@author: Una
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:08:09 2018

@author: Una
"""

# Useful starting lines
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#load_ext autoreload
#autoreload 2

from proj1_helpers import *
from lib_dataPreprocessing import *
from lib_errorMetrics import *
from lib_MLmodels import *

import os
import matplotlib.pyplot as plt
import json
import seaborn as sns


#SETUP 
plottingOn=0
createNew999Features=0
createNewDegreeFeatures=0
createNewQuadraticFeatures=1
degree=5

leastSquaresOn=0
gradientDescentOn=0
ridgeRegressionOn=0
logisticRegressionOn=1
logisticRegressionRegularizedOn=0

#-------------------------------------------------------------------------------

# DATA LOADING - TRAINING 
data_path = os.path.abspath("../data/train.csv")
yb_train, x_train, ids_train= load_csv_data(data_path, sub_sample=False)
# Check types of labels
print(set(yb_train))
print('Number of trials training:', len(yb_train))


#-------------------------------------------------------------------------------
#FEATURE PROCESSING 
# Labels from [-1,1] to [0,1]
yb_train=y_to_01(yb_train)
# augumenting features 
x_train_augumented=featuresPreprocessing(x_train,createNew999Features,createNewQuadraticFeatures,createNewDegreeFeatures,degree,percentileOutliers,plottingOn)

num_features= len(x_train_augumented[1,:])
num_samples = len(yb_train)
samples = np.arange(num_samples)

#-------------------------------------------------------------------------------
#BUILDING MODEL 
    
# LEAST SQUARES
if (leastSquaresOn==1):
    typeNormLog='normal'
    w, RMSE=least_squares(yb_train, x_train_augumented)
    print('LEAST SQUARES:')
    print('--RMSE:', RMSE)


## GRADIENT DESCENT 
if (gradientDescentOn==1):
    max_iters =500
    gamma =0.05 # 0.7
    typeNormLog='normal'
    w, RMSE=gradient_descent(yb_train,x_train_augumented,max_iters,gamma,plottingOn)
    print('GRADIENT DESCENT:')
    print('--RMSE:',RMSE)
    

## RIDGE REGRESSION 
if (ridgeRegressionOn==1):
    lambdas = np.logspace(-4, 0, 30)
    typeNormLog='normal'
    w, RMSE, lambdaDecision= optimal_ridge_regression(yb_train,x_train_augumented, lambdas,plottingOn)
    print('RIDGE REGRESSION:')
    print('--RMSE:',RMSE)
    print('--lambda:', lambdaDecision)

## LOGISTIC REGRESSION 
if (logisticRegressionOn==1):
    max_iters = 3000#30000
    gamma = 0.1
    initial_w = np.zeros(np.shape(features[1,:]))
    typeNormLog='logistic'
    w, loss= logistic_regression(yb_train, x_train_augumented, initial_w, max_iters, gamma)
    print('LOGISTIC REGRESSION :')
    print('--loss:',loss)

## PENALIZED LOGISTIC REGRESSION 
if (logisticRegressionRegularizedOn==1):
    max_iters = 10000
    gamma =0.1 # 0.00001 
    lambda_= 0.5
    initial_w = np.zeros(np.shape(features[1,:]))
    typeNormLog='logistic'
    w, loss= reg_logistic_regression(yb_train, x_train_augumented, lambda_, initial_w, max_iters, gamma)
    print('REGULARIZED LOGISTIC REGRESSION :')
    print('--loss:',loss)
    
#-------------------------------------------------------------------------------
# TRANSFORMING LABELS + STATISTICS
#transform labels to classes 
y_pred_train=x_train_augumented.dot(w)
y_pred_train_cl=predictionToClasses(y_pred_train,typeNormLog)

#calculate statistics 
error_train, accuracy_train=calcResultsStatistics(yb_train,y_pred_train_cl) 

#-------------------------------------------------------------------------------
# PREDICTING TEST DATA

# DATA LOADING  - TEST
data_path = os.path.abspath("../data/test.csv")
yb_test, x_test, ids_test= load_csv_data(data_path, sub_sample=False)
# Check types of labels
print(set(yb_test))
print('Number of trials test :', len(yb_test))

# FEATURE PREPROCESSING
# augumenting features 
x_test_augumented=featuresPreprocessing(x_test,createNew999Features,createNewQuadraticFeatures,createNewDegreeFeatures,degree,percentileOutliers,plottingOn)

#transform labels to classes 
y_pred_test=x_test_augumented.dot(w)
y_pred_test_cl=predictionToClasses(y_pred_test,typeNormLog)

#CREATING OUTPUT FILE FOR SUBMISSION 
name_file='predictionLabels.csv'
create_csv_submission(ids_test, y_pred_test_cl, name_file)

#-------------------------------------------------------------------------------
#STORING WEIGHTS 
wList=w.tolist()
print(wList)

with open("weights.json", "w") as write_file:
    json.dump(wList, write_file)
    
with open("weights.json", "r") as read_file:
    wList = json.load(read_file)
weights=np.asarray(wList)
print(weights)