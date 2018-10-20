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
degree=2

leastSquaresOn=1
gradientDescentOn=0
ridgeRegressionOn=0

numFolds=10
seed=1

# DATA LOADING 
data_path = os.path.abspath("../data/train.csv")
yb, x, ids= load_csv_data(data_path, sub_sample=False)

# -999values
x=resolve999values(x,createNew999Features,plottingOn)
 
# removing outliers
percentileOutliers=96
x=removeOutliers(x, percentileOutliers)
    
# building polinomial features
x=build_poly(x,createNewDegreeFeatures,createNewQuadraticFeatures, degree)

#normalizing data
#x_norm, mean_x, std_x= standardize01(x)
x_norm, mean_x, std_x= standardize(x)

# addind 1 for firs column
x_norm=augument_feature(x_norm)

#build folds 
num_features= len(x[1,:])
num_trials=len(yb)
k_indices = build_k_indices(num_trials, numFolds, seed)


#evaluate model using k fold crossvalidation
RMSE_tr_all=[]
RMSE_te_all=[]
percDiff_tr_all=[]
percDiff_te_all=[]
for k in range(numFolds):
    
    #test set
    x_test=x_norm[k_indices[k,:]]
    y_test=yb[k_indices[k,:]]
    
    #train set
    ind_train=[]
    for i in k_indices:
        if not i in k_indices[k,:]:
            ind_train=np.append(ind_train,i)
    ind_train=ind_train.astype(int)
    x_train=x_norm[ind_train]
    y_train=yb[ind_train]
    
    
    if (leastSquaresOn==1):
        w, loss_tr=least_squares(y_train, x_train)            
    if (gradientDescentOn==1):
        max_iters =500
        gamma =0.05 # 0.7
        w, loss_tr=gradient_descent(y_train,x_train,max_iters,gamma,plottingOn)
        
    #saving loss  
    loss_te=loss_rmse(y_test, x_test, w)
    RMSE_tr_all=np.append(RMSE_tr_all,loss_tr)
    RMSE_te_all=np.append(RMSE_te_all,loss_te) 
    #differences in labels
    y_pred_tr=x_train.dot(w)
    y_pred_tr_1=predictionsToClasses(y_train,y_pred_tr,plottingOn)
    y_pred_te=x_test.dot(w)
    y_pred_te_1=predictionsToClasses(y_test,y_pred_te,plottingOn)
    #calculate statistics 
    percDiff_tr=calcResultsStatistics(y_train,y_pred_tr_1)  
    percDiff_te=calcResultsStatistics(y_test,y_pred_te_1)    
    percDiff_tr_all=np.append(percDiff_tr_all,percDiff_tr)
    percDiff_te_all=np.append(percDiff_te_all,percDiff_te)
    
RMSE_tr=np.mean(RMSE_tr_all)
RMSE_te=np.mean(RMSE_te_all)
RMSE_tr_var=np.std(RMSE_tr_all)
RMSE_te_var=np.std(RMSE_te_all)
percDiff_tr=np.mean(percDiff_tr_all)
percDiff_te=np.mean(percDiff_te_all)
percDiff_tr_var=np.std(percDiff_tr_all)
percDiff_te_var=np.std(percDiff_te_all)

print('*****')
print('RMSE_tr:',RMSE_tr)
print('RMSE_te:',RMSE_te)
print('percDiff_tr:',percDiff_tr)
print('percDiff_te:',percDiff_te)