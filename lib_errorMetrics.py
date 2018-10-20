# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:47:06 2018

@author: Una
"""
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
                elif int(y_true[i]) == (-1):
                        if (y_pred[i]==1):
                                conf_mat[0][1] = conf_mat[0][1] +1 #FP
                        else:
                                conf_mat[1][1] = conf_mat[1][1] +1 #TN

        accuracy = float(conf_mat[0][0] + conf_mat[1][1])/(len(y_true))
        precision = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[0][1])  
        sensitivity = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[1][0])
        specificity = float(conf_mat[1][1])/float(conf_mat[1][1] + conf_mat[1][0]) 
        return conf_mat, accuracy, precision, sensitivity, specificity 
    
    
# PREDICTIONS (FLOAT) TO CLASSES (-1,1)
def predictionsToClasses(yb,y_predicted,plottingOn):
    indx1=np.where(yb==1)
    indx_1=np.where(yb==-1)
    
    # predicted label to -1 and 1
    y_predicted_1=np.copy(y_predicted)
    y_predicted_1[np.where(y_predicted>0)]=1
    y_predicted_1[np.where(y_predicted<0)]=-1
    
    # visualizinG true and predicted labels
    if (plottingOn==1):
        fig, ax1 = plt.subplots(figsize=(50,25), dpi= 40)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(indx1,y_predicted_1[indx1],c='b')
        ax1.scatter(indx_1,y_predicted_1[indx_1],c='r')
        ax1.set(xlabel='y label', ylabel='predicted label',title='True vs predicted labels')
        ax1.grid()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(indx1,y_predicted_1[indx1],c='b')
        ax2.set(xlabel='y label', ylabel='predicted label',title=' True label = 1')
        ax2.grid()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(indx_1,y_predicted_1[indx_1],c='r')
        ax3.set(xlabel='y label', ylabel='predicted label',title='True label = -1')
        ax3.grid()
        fig.savefig("true_vs_predicted_labels_1")
    return y_predicted_1


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
    
    return percDifferences