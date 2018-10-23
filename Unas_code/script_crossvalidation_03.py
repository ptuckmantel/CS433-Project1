# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:15:47 2018

@author: Una
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:28:36 2018

@author: Una
"""


# Useful starting lines
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#load_ext autoreload
#autoreload 2

from proj1_helpers import *
from functions import * 
import os
import matplotlib.pyplot as plt
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

percentileOutliers=96

numFolds=4
seed=1

max_iters = 3000
gamma =0.1 # 0.00001 
lambda_= 0.5
initial_w = np.zeros(np.shape(features[1,:]))
        
#-------------------------------------------------------------------------------

# DATA LOADING  - TEST
data_path = os.path.abspath("../data/test.csv")
yb_test, x_test, ids_test= load_csv_data(data_path, sub_sample=False)
# Check types of labels
print(set(yb_test))
print('Number of trials test :', len(yb_test))

# DATA LOADING - TRAINING 
data_path = os.path.abspath("../data/train.csv")
yb_train, x_train, ids_train= load_csv_data(data_path, sub_sample=False)
# Check types of labels
print(set(yb_train))
print('Number of trials training:', len(yb_train))


#----------------------------------------------------------------------------------
#FEATURE PROCESSING 
# Labels from [-1,1] to [0,1]
yb_train=y_to_01(yb_train)
# augumenting features 
x_train_augumented=featuresPreprocessing(x_train,createNew999Features,createNewQuadraticFeatures,createNewDegreeFeatures,degree,percentileOutliers,plottingOn)

num_features= len(x_train_augumented[1,:])
num_samples = len(yb_train)
samples = np.arange(num_samples)

#------------------------------------------------------------------------------
# BUILDING FOLDS  
num_features= len(x[1,:])
num_trials=len(yb)
k_indices = build_k_indices(num_trials, numFolds, seed)


#------------------------------------------------------------------------------
# CROSSVALIDATION 
#evaluate model using k fold crossvalidation
loss_tr_all=[]
loss_te_all=[]
error_tr_all=[]
error_te_all=[]
accuracy_tr_all=[]
accuracy_te_all=[]
for k in range(numFolds):
    
    #test set
    x_CV_te=x_norm[k_indices[k,:]]
    y_CV_te=yb_train[k_indices[k,:]]
    
    #train set
    ind_tr=[]
    for i in k_indices:
        if not i in k_indices[k,:]:
            ind_tr=np.append(ind_tr,i)
    ind_tr=ind_tr.astype(int)
    x_CV_tr=x_norm[ind_tr]
    y_CV_tr=yb_train[ind_tr]
    
    
    if (logisticRegressionOn==1):   
        w, loss_tr= logistic_regression(y_CV_tr, x_CV_tr, initial_w, max_iters, gamma)
        loss_te=loss_maxL(y_CV_te, x_CV_te, w)
        typeNormLog='logistic'

    if (logisticRegressionRegularizedOn==1):
        w, loss_tr= reg_logistic_regression(y_CV_tr, x_CV_tr, lambda_, initial_w, max_iters, gamma)
        loss_te=loss_maxL(y_CV_te, x_CV_te, w)
        typeNormLog='logistic'
        
    if (leastSquaresOn==1):
        w, loss_tr=least_squares(y_CV_tr, x_CV_tr) 
        loss_te=loss_rmse(y_CV_te, x_CV_te, w)
        typeNormLog='normal'
           
    if (gradientDescentOn==1):
        max_iters =500
        gamma =0.05 # 0.7
        w, loss_tr=gradient_descent(y_CV_tr,x_CV_tr,max_iters,gamma,plottingOn)
        loss_te=loss_rmse(y_CV_te, x_CV_te, w)
        typeNormLog='normal'
        
    #saving loss  
    loss_tr_all=np.append(loss_tr_all,loss_tr)
    loss_te_all=np.append(loss_te_all,loss_te) 
    #transform labels to classes 
    y_pred_tr=x_CV_tr.dot(w)
    y_pred_tr_cl=predictionToClasses(y_pred_tr,typeNormLog)
    y_pred_te=x_CV_test.dot(w)
    y_pred_te_cl=predictionToClasses(y_pred_te,typeNormLog)
    #calculate statistics 
    error_tr, accuracy_tr=calcResultsStatistics(y_CV_tr,y_pred_tr_cl)  
    error_te, accuracy_te=calcResultsStatistics(y_CV_te,y_pred_te_cl)    
    error_tr_all=np.append(error_tr_all,error_tr)
    error_te_all=np.append(error_te_all,error_te)
    accuracy_tr_all=np.append(accuracy_tr_all,accuracy_tr)
    accuracy_te_all=np.append(accuracy_te_all,accuracy_te)
    
    print()
    print('k:', k)
    print('TRAINING-> loss:',loss_tr, 'error:', error_tr, 'accuracy:', accuracy_tr)
    print('TESTING-> loss:',loss_te, 'error:', error_te, 'accuracy:', accuracy_te)

#------------------------------------------------------------------------------
# PERFORMANCE METRICS
    
loss_tr=np.mean(loss_tr_all)
loss_te=np.mean(loss_te_all)
loss_tr_var=np.std(loss_tr_all)
loss_te_var=np.std(loss_te_all)
error_tr=np.mean(error_tr_all)
error_te=np.mean(error_te_all)
error_tr_var=np.std(error_tr_all)
error_te_var=np.std(error_te_all)
accuracy_tr=np.mean(accuracy_tr_all)
accuracy_te=np.mean(accuracy_te_all)
accuracy_tr_var=np.std(accuracy_tr_all)
accuracy_te_var=np.std(accuracy_te_all)

print()
print('***** MEAN VALUES ')
print('loss_tr:',loss_tr)
print('loss_te:',loss_te)
print('error_tr:',error_tr)
print('error_te:',error_te)
print('accuracy_tr:',accuracy_tr)
print('accuracy_te:',accuracy_te)

