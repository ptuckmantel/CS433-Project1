# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:47:06 2018

@author: Una
"""
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from lib_dataPreprocessing import *
from lib_MLmodels import *

# CONFUSION MATRIX
def conf_mat(y_pred, y_true):
    '''calculates confusion matrix for given true and predicted labels
    assumes that labels are 0 and 1'''
    # confusion matrix
    conf_mat = [[0, 0], [0, 0]]

    for i in range(len(y_pred)):
            if int(y_true[i]) == 1: 
                    if (y_pred[i]==1):
                            conf_mat[0][0] = conf_mat[0][0] + 1 #TP
                    else:
                            conf_mat[1][0] = conf_mat[1][0] + 1 #FN
            elif int(y_true[i]) == (0): ## ???
                    if (y_pred[i]==1):
                            conf_mat[0][1] = conf_mat[0][1] +1 #FP
                    else:
                            conf_mat[1][1] = conf_mat[1][1] +1 #TN

    accuracy = float(conf_mat[0][0] + conf_mat[1][1])/(len(y_true))
    precision = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[0][1]) 
    sensitivity = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[1][0])
    specificity = float(conf_mat[1][1])/float(conf_mat[1][1] + conf_mat[1][0]) 
    return conf_mat, accuracy, precision, sensitivity, specificity 
    

#
#def predictionToClassesLogistic(y):
#    '''transforms probability values to 0 or 1'''
#    y=y-0.5
#    y=np.ceil(y)
#    return y

def predictionToClasses(y_predicted,typeNormLog):
    '''transforms 0,1 labels  to -1,1, 
    different transform if logistic regression - then sigmoid needed'''
    if (typeNormLog=='logistic'):
        y_out=sigmoid(y_predicted)
        y_out=y_out-0.5
        y_out=np.ceil(y_out)
    if (typeNormLog=='normal'):
        y_out=y_predicted-0.5
        y_out=np.ceil(y_out)
    return y_out


# CALCULATE ALL STATISTICS 
def calcResultsStatistics(yb,y_predicted_1):
    '''calculates statistics of performance
    counts number of different predictions, confusion matrix, 
    accuracy, precision, sensitivity, specificity
    prints all, but returns only percentge of different ones and accuracy '''
    # count number of different ones 
    indxDiff=np.where(yb!=y_predicted_1)
    indxDiff2=np.asarray(indxDiff)[0,:]
    print('Number different predictions:', len(indxDiff2))
    print('Percentage different ones:', len(indxDiff2)/len(yb))
    percDifferences=len(indxDiff2)/len(yb)
    #confusion matrix and error metrics
    conf_matrix, accuracy, precision, sensitivity, specificity = conf_mat(y_predicted_1, yb)
    print('Confusion matrix:', conf_matrix)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    return percDifferences, accuracy

def validate_model(w,x_train_augumented,yb_train, x_test_augumented, yb_test, typeNormLog): 
    '''used for crossvalidation
    given train and test feature, labels and weights performs:
    -predicts labels for test set, transforms to right labels and calculates error and accuracy for train
    -predicts labels for test set, transforms to right labels and calculates error and accuracy for test
    -prints results
    -returns error and accuracy both for train and test '''    
    # TRAIN SET 
    print('------------------------------: ')
    print('TRAINING SET RESULTS: ')
    #transform labels to classes 
    y_pred_train=x_train_augumented.dot(w)
    y_pred_train_cl=predictionToClasses(y_pred_train,typeNormLog)
    #calculate statistics 
    error_train, accuracy_train=calcResultsStatistics(yb_train,y_pred_train_cl)
    # TEST SET 
    print('TEST SET RESULTS: ')
    #transform labels to classes 
    y_pred_test=x_test_augumented.dot(w)
    y_pred_test_cl=predictionToClasses(y_pred_test,typeNormLog)
    #calculate statistics 
    error_test, accuracy_test=calcResultsStatistics(yb_test,y_pred_test_cl)
    return error_train, accuracy_train, error_test, accuracy_test
