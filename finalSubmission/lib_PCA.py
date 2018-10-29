# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:59:50 2018

@author: Una
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_PCA_reduction(x_train, x_test,PCAvarianceThreshold):
    pca=PCA(PCAvarianceThreshold)
    pca2=pca.fit(x_train)
    pca_train=pca.transform(x_train)
    pca_test = pca.transform(x_test)
    var=pca.explained_variance_ratio_
    # print('x:',np.shape(x))
    # print('pca2:',np.shape(pca2))
    # print('pca3:',np.shape(pca3))
    # print('var:',var)
    print('Number of features after PCA:', np.shape(pca_train))
    return pca_train, pca_test
    # pca = PCA().fit(x)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance');
    #
    # variance=pca.explained_variance_ratio_
    # print('variance:',variance)
    # indxOrder=numpy.argsort(variance)
    # print('indx order:',indxOrder)
    # varOrdered=var[indxOrder]
    # print('variance ordered:',varOrdered)
    # varCumsum=np.cumsum(varOrdered)
    # print('variance ordered:',varOrdered)
    # indxToInclude=varCumsum<PCAvarianceThreshold
    # print('indxToInclude:', indxToInclude)
    # x=x[:,indxToInclude]
    # print('shape x:', np.shape(x))
    # x=np.squeeze(x)
    # print('shape x:', np.shape(x))
    
#    pca = PCA(n_components=2)
#    pca.fit(X)
#    X_pca = pca.transform(X)
#    print("original shape:   ", X.shape)
#    print("transformed shape:", X_pca.shape)
    