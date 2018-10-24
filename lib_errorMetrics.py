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

# PARAMETERS THAT ARE NEEDED AT SOME POINT FOR THIS FUNCTIONS 
# best to specify them at the beginning of the main scrip 
#plottingOn=0

# CONFUSION MATRIX
def conf_mat(y_pred, y_true):
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
    


def predictionToClassesLogistic(y):
    y=y-0.5
    y=np.ceil(y)
    return y

def predictionToClasses(y_predicted,typeNormLog):
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