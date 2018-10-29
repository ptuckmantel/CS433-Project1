# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 00:38:17 2018

@author: Una

BUILDING MODEL WITH CERTAIN PARAMETERS AND THEN PREDICTING ON TEST DATA 
CREATES OUTPUT FILE FOR SUBMITION ON KAGGLE 
trains model on training set 
prints both accuracies and errors for training set 
stores weights in 'weights.json' file 
loads test set and using these weights computes predictions
stores predictions in 'predictionLabels.csv'
SEPARATES TEST SET IN TWO PARTS AND PREDICTS LABELS FOR BOTH AND THEN CONCATENATES
THIS IS BECAUSE SOME COMPUTERS DONT HAVE ENOUGH MEMORY
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
from lib_PCA import *

import os
import matplotlib.pyplot as plt
import json
#import seaborn as sns


#SETUP 
params ={
  "plottingOn": 0,
  "remove999FeaturesOn": 0,
  "fraction999Threshold": 0.8,
  "createNew999Features": 1, 
  "createNewDegreeFeatures":1,
  "createNewQuadraticFeatures":1,
  "degree":10,
  "addSinCosFeatures":1,
  "toGaussDistributionOn":0,
  "skewnessThreshold":0.5,
  "removeOutliersOn": 1,
  "numSigmas": 3,
  "performPCA": 0,
  "PCAvarianceThreshold": 0.99,
  "bucketing0n": 1,
  "numBucketsPerFeature": 10,
  "addEtaThetaFeatures":1,
  "typeModel": 5,  #1=leastSquares, 2=gradientDescent, 3=ridgeRegression, 4=logisticRegression, 5=logisticRegressionRegularized
  "ratioTrain": 0.8,
  "max_iters": 5000,
  "gamma":0.1,
  "lambda": 0.0001
}

##-------------------------------------------------------------------------------

# DATA LOADING - TRAINING 
data_path = os.path.abspath("../data/train.csv")
yb_train, x_train, ids= load_csv_data(data_path, sub_sample=False)
print('Number of trials training:', len(yb_train))

#FEATURE PROCESSING 
# Labels from [-1,1] to [0,1]
yb=y_to_01(yb_train)

# augumenting features 
x_train_augumented=featuresPreprocessing2(x_train,params)

    
#BUILDING MODEL 
w, loss, typeNormLog = train_model(x_train_augumented, yb, params["typeModel"], params)  

# VALIDTE MODEL
y_pred_train=x_train_augumented.dot(w)
y_pred_train_cl=predictionToClasses(y_pred_train,typeNormLog)
y_pred_train_11=2*(y_pred_train_cl-0.5)
error_train, accuracy_train=calcResultsStatistics(yb_train,y_pred_train_11) 
print('errTr:',error_train, 'accTr:',accuracy_train)

# STORING WEIGHTS
with open("weights.json", "w") as write_file:
    json.dump(list(w), write_file)

#-------------------------------------------------------------------------------

'''TEST SET'''


NUM_PARTS=2
y_pred_test_total=[]

# DATA LOADING  - TEST
data_path = os.path.abspath("../data/test.csv")
yb_test, x_test, ids_test= load_csv_data(data_path, sub_sample=False)
print('Number of trials test :', len(yb_test))

numPointsPerPart=int(len(yb_test)/NUM_PARTS)
for p in range(1,NUM_PARTS+1):
    if (p==NUM_PARTS):
#        yb_test_iter=yb_test[(p-1)*numPointsPerPart:]
        x_test_iter=x_test[(p-1)*numPointsPerPart:,:]
#        ids_test_iter=ids_test[(p-1)*numPointsPerPart:,:]
    else: 
#        yb_test_iter=yb_test[(p-1)*numPointsPerPart:p*numPointsPerPart]
        x_test_iter=x_test[(p-1)*numPointsPerPart:p*numPointsPerPart,:]
#        ids_test_iter=ids_test[(p-1)*numPointsPerPart:p*numPointsPerPart,:]
        
    # PROCESS FEATURES 
    x_test_iter=featuresPreprocessing2(x_test_iter,params)
    
    # LOAD WEIGHTS
    with open("weights.json", "r") as read_file:
        wList = json.load(read_file)
    weights = np.asarray(wList)
    
    # PREDICT AND STORE RESULTS
    #transform labels to classes
    y_pred_test_iter=x_test_iter.dot(weights)
    y_pred_test_iter_cl=predictionToClasses(y_pred_test_iter,typeNormLog)
    y_pred_test_iter_11=2*(y_pred_test_iter_cl-0.5)
    
    y_pred_test_total=np.append(y_pred_test_total,y_pred_test_iter_11)

#CREATING OUTPUT FILE FOR SUBMISSION
name_file='predictionLabels.csv'
create_csv_submission(ids_test, y_pred_test_total, name_file)

print('Done!')

