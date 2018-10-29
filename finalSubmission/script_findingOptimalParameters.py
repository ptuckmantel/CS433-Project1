# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 00:38:17 2018

@author: Una


FOR FINDING BEST PARAMETERS SETUP 
uses dictionary for storing parameters 
using 80% training for training and 20% for testing 
storing results in testingDiffParams.json file 
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
  "skewnessThreshold":0,
  "removeOutliersOn": 1,
  "numSigmas": 3,
  "performPCA": 0,
  "PCAvarianceThreshold": 0.99,
  "bucketing0n": 1,
  "numBucketsPerFeature": 3,
  "addEtaThetaFeatures":0,
  "typeModel": 5,  #1=leastSquares, 2=gradientDescent, 3=ridgeRegression, 4=logisticRegression, 5=logisticRegressionRegularized
  "ratioTrain": 0.8,
  "max_iters": 1000,
  "gamma":0.1,
  "lambda": 0.001
}

errTrainAll=[]
errTestAll=[]
accTrainAll=[]
accTestAll=[]
weightsAll=[]

allSetups=np.array([[0,0.6,0,0,1,3,0,10,1,0,0,0,0,3,0],
                    [1,0.6 ,0,0,1,3,0,10,1,0,0,0,0,3,0], #removing 999 features 
                    [1,0.25,0,0,1,3,0,10,1,0,0,0,0,3,0], #removing 999 features
                    [0,0.6, 0,0,1,3,1, 2,1,0,0,0,0,3,0], #degree features 
                    [0,0.6, 0,0,1,3,1, 5,1,0,0,0,0,3,0], #degree features 
                    [0,0.6, 0,0,1,3,1,10,1,0,0,0,0,3,0], #degree features    
                    [0,0.6, 0,0,1,3,1,10,1,0,1,0,0,3,0], #sincos features
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,0,3,0], #new 999 features
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,1,3,0], #bucketing 
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,1,5,0], #bucketing 
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,1,10,0], #bucketing 
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,1,15,0], #15 bucketing
                    [0,0.6, 0,0,1,3,1,10,1,1,1,0,1,10,1], #theta eta #best so far 
                    [0,0.6, 0,0,1,5,1,10,1,1,1,0,1,10,1], #removing outliers 5sigma
                    [0,0.6, 0,0,0,3,1,10,1,1,1,0,1,10,1], #no removing outliers
                    [0,0.6, 1,0,1,3,1,10,1,1,1,0,1,10,1], #gauss, threshold=0
                    [0,0.6, 1,0.5,1,3,1,10,1,1,1,0,1,10,1], #gauss, threshold=0  
                    [0,0.6, 0,0,1,3,1,10,1,1,1,1,1,10,1]]) #PCA

  

#-------------------------------------------------------------------------------

# DATA LOADING - TRAINING 
data_path = os.path.abspath("../data/train.csv")
yb, x, ids= load_csv_data(data_path, sub_sample=False)
print('Number of trials training:', len(yb))

#-------------------------------------------------------------------------------
#FEATURE PROCESSING 
# Labels from [-1,1] to [0,1]
yb=y_to_01(yb)

for i in range(0,len(allSetups)):
    params["remove999FeaturesOn"]=allSetups[i][0]
    params["fraction999Threshold"]=allSetups[i][1]
    params["toGaussDistributionOn"]=allSetups[i][2]
    params["skewnessThreshold"]=allSetups[i][3]
    params["removeOutliersOn"]=allSetups[i][4]
    params["numSigmas"]=allSetups[i][5]
    params["createNewDegreeFeatures"]=allSetups[i][6]
    params["degree"]=int(allSetups[i][7])
    params["createNewQuadraticFeatures"]=allSetups[i][8]
    params["createNew999Features"]=allSetups[i][9]
    params["addSinCosFeatures"]=allSetups[i][10]
    params["performPCA"]=allSetups[i][11]
    params["bucketing0n"]=allSetups[i][12]
    params["numBucketsPerFeature"]=int(allSetups[i][13])
    params["addEtaThetaFeatures"]=allSetups[i][14]
    
    xi=np.copy(x)
    
    # augumenting features 
    x_augumented=featuresPreprocessing2(xi,params)
    
    # SPLITTING DATA TO TRAIN AND TEST SET 
    x_train_augumented, yb_train, x_test_augumented, yb_test= split_data(x_augumented, yb, params["ratioTrain"], seed=1)
    
    #PCA
    if (params["performPCA"]==1):
         x_train_augumented, x_test_augumented =perform_PCA_reduction(x_train_augumented,x_test_augumented, params["PCAvarianceThreshold"])
    
    
    num_features= len(x_train_augumented[1,:])
    num_samples = len(yb_train)
    samples = np.arange(num_samples)
    
        
    #BUILDING MODEL 
    w, loss, typeNormLog = train_model(x_train_augumented, yb_train, params["typeModel"], params)  
    
    # VALIDTE MODEL
    error_train, accuracy_train, error_test, accuracy_test= validate_model(w,x_train_augumented,yb_train, x_test_augumented, yb_test, typeNormLog)
    
    print('STEP ',i,':   errTr:',error_train, 'errTe:', error_test, 'accTr:',accuracy_train, 'accTe:', accuracy_test)
    
    # STORE DATA
    weightsAll=np.append(weightsAll,w)
    errTrainAll=np.append(errTrainAll,error_train)
    errTestAll=np.append(errTestAll,error_test)
    accTrainAll=np.append(accTrainAll,accuracy_train)
    accTestAll=np.append(accTestAll,accuracy_test)
    
    stats={
        "errTrain":list(errTrainAll),
        "errTest":list(errTestAll),
        "accTrain":list(accTrainAll),
        "accTest":list(accTestAll),
        "weightsAll":list(weightsAll)
    }
        
    with open("testingDiffParams.json", "w") as write_file:
        json.dump(stats, write_file)


print('Done!')

