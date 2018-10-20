# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:31:57 2018

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
createNewDegreeFeatures=1
createNewQuadraticFeatures=0
degree=5

leastSquaresOn=1
gradientDescentOn=0
ridgeRegressionOn=0

# DATA LOADING 
data_path = os.path.abspath("../data/train.csv")
yb, x, ids= load_csv_data(data_path, sub_sample=False)

# Check types of labels
set(yb)
print('Number of trials:', len(yb))
yTrue=np.asarray(np.where(yb==1))
yFalse=np.asarray(np.where(yb==(-1)))
num_yTrue=len(yTrue[0,:])
num_yFalse=len(yFalse[0,:])
ratioTF=num_yTrue/num_yFalse
print('Ratio of 1 over -1 labels:', ratioTF)

num_features= len(x[1,:])
num_samples = len(yb)
samples = np.arange(num_samples)

#Plotting original data
if (plottingOn==1):
    fig=plt.figure(figsize=(50, 25), dpi= 40)
    for i in range(num_features):
        ax1 = fig.add_subplot(6, 5, i+1)
        ax1.scatter(samples, x[:,i],marker='.');
        ax1.set(xlabel='sample', ylabel='feature value',title='Feature '+ str(i+1))
        ax1.grid()
    fig.savefig("input_features")
    
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

# plotting normalized and without outliers data
if (plottingOn==1):
    fig=plt.figure(figsize=(50, 25), dpi= 40)
    for i in range(num_features):
        ax1 = fig.add_subplot(6, 5, i+1)
        ax1.scatter(samples, x[:,i],marker='.');
        ax1.set(xlabel='sample', ylabel='feature value',title='Feature '+ str(i+1))
        ax1.grid()
    fig.savefig("input_features_without999_normalized_withoutOutliers")
    
#plotting data distribution 
if (plottingOn==1):
    fig, ax1 = plt.subplots(figsize=(50,25), dpi= 40)
    ax1.set_title('Distibution of data')
    ax1.boxplot(x_norm)
    ax1.grid()
    fig.savefig("data_distibution")

#------------------------------------------------------------------------------------------------------
features=x_norm
    
# LEAST SQUARES
if (leastSquaresOn==1):
    w, RMSE=least_squares(yb, features)
    #RMSE_ls=loss_rmse(yb, features, w_ls)
    print('LEAST SQUARES:')
    print('--RMSE:', RMSE)


## GRADIENT DESCENT 
if (gradientDescentOn==1):
    max_iters =500
    gamma =0.05 # 0.7
    
    w, RMSE=gradient_descent(yb,features,max_iters,gamma,plottingOn)
    print('GRADIENT DESCENT:')
    print('--RMSE:',RMSE)
    

## RIDGE REGRESSION 
if (ridgeRegressionOn==1):
    lambdas = np.logspace(-4, 0, 30)
    w, RMSE, lambdaDecision= optimal_ridge_regression(yb,features, lambdas,plottingOn)
    print('RIDGE REGRESSION:')
    print('--RMSE:',RMSE)
    print('--lambda:', lambdaDecision)


#--------------------------------------------------------------------------------------------------    

#ADDAPTING OUTPUT TO HAVE ONLY -1 and 1
y_predicted=features.dot(w)
y_predicted_1=predictionsToClasses(yb,y_predicted,plottingOn)

#calculate statistics 
calcResultsStatistics(yb,y_predicted_1)