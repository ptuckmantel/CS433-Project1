# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:45:24 2018

@author: Una
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

from proj1_helpers import *
from lib_errorMetrics import *
from lib_MLmodels import *

# PARAMETERS THAT ARE NEEDED AT SOME POINT FOR THIS FUNCTIONS 
# best to specify them at the beginning of the main scrip 
#plottingOn=0
#createNew999Features=0
#createNewDegreeFeatures=0
#createNewQuadraticFeatures=1
#degree=2

# SPLITING DATA
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing  """
    np.random.seed(seed)
    indx=np.random.permutation(len(y))
    x_rand=x[indx]
    y_rand=y[indx]
    indxMax=int(ratio*len(y))
    x_train=x_rand[:indxMax]
    y_train=y_rand[:indxMax]
    x_test=x_rand[indxMax:]
    y_test=y_rand[indxMax:]
    return x_train, y_train, x_test, y_test


# DATA NORMALIZATION 
def standardize01(x):
    """standardize the original data set to values between 0 and 1"""
    min_x = np.amin(x, axis=0)
    max_x = np.amax(x, axis=0)
    x_norm = (x - min_x)/(max_x-min_x)
    std_x = np.std(x_norm,axis=0)
    mean_x = np.mean(x_norm,axis=0)
    return x_norm, mean_x, std_x

def standardize(x):
    """standardize the original data set so that mean= and std=1"""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x

# GAUSSIAN DISTRIBUTION 
def cube_transform(x):
    cubed_x = x**3
    return cubed_x

def log_transform(x):
    shifted_x = x - np.min(x) + 1
    return np.log(shifted_x)
    
def transform_to_gauss(x,skewnessThreshold):
    '''transform data so that distribution is more Gauss-like
    in case skewness is negative perform cube transform and in case
    skewness is positive perform log transform
    possible to set up threshold for skewness, typicaly 0.5'''
    corrected_data=[]
    D = x.shape[1]
    for d in range(D):
        skewness = skew(x[:, d])
        if skewness < -skewnessThreshold:
            corrected_data.append(cube_transform(x[:, d]))
        elif skewness > skewnessThreshold:
            corrected_data.append(log_transform(x[:, d]))
        else:
            corrected_data.append(x[:, d])
    return np.array(corrected_data).T
     

# Y LABELS TO 0 AND 1
def y_to_01(y):
    '''transforms labels from {-1,1} to {0.1} '''
    y[y==-1]=0
    return y

# BUILD POLINOMIAL BASE
def build_poly_degree(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree
    - does the same for eaxh feature, but dont combine them (no interaction terms)
    - returns new features that should be appended to the original ones"""
    numOriginalFeatures=len(x[0,:])
    newFeatures=np.zeros([len(x[:,0]),((degree-1)*numOriginalFeatures)])
    indx=0
    for i in range(numOriginalFeatures):
        for j in range(2,degree+1):
            newFeatures[:,indx]=np.power(x[:,i],j)
            indx=indx+1
    return newFeatures

def build_quadratic_poly(x):
    """polynomial basis functions with degree=2 but with all possible recombinations of features
    - returns new features that should be appended to the original ones"""
    numOriginalFeatures=len(x[0,:])
    numNewFeatures=np.int_((numOriginalFeatures*(numOriginalFeatures+1))/2)
    newFeatures=np.zeros([len(x[:,0]),numNewFeatures])
    indx=0
    for i in range(numOriginalFeatures):
        for j in range(0,i+1):
            newFeatures[:,indx]=x[:,i]*x[:,j]
            indx=indx+1   
    return newFeatures

def build_poly(x,createNewDegreeFeatures,createNewQuadraticFeatures, degree):
    ''' depending on setup performs either only degree expansion or degree=2 with interaction terms 
    - returns initial features together with new polynomial ones appended at the end''' 
    if (createNewDegreeFeatures==1):
        newFeatures1=build_poly_degree(x, degree)
    if (createNewQuadraticFeatures==1):
        newFeatures2=build_quadratic_poly(x)
    if (createNewDegreeFeatures==1):
        x=np.column_stack((x,newFeatures1))
    if (createNewQuadraticFeatures==1):
        x=np.column_stack((x,newFeatures2))
    return x

def build_sincos_features(x):
    ''' builds new sin and cos features 
    - returns new features that should be appended to the orignal ones'''
    numOriginalFeatures=len(x[0,:])
    newFeatures=np.zeros([len(x[:,0]),(2*numOriginalFeatures)])
    indx=0
    for i in range(numOriginalFeatures):
        newFeatures[:,indx]=np.sin(x[:,i])
        newFeatures[:,indx+1] =np.cos(x[:, i])
        indx=indx+2
    return newFeatures

# AUTUMENTING FEATURES
def augument_feature(x):
    """add column of ones at the beginning """
    col1=np.ones([len(x[:,0]),1])
    x=np.column_stack((col1,x))
    return x


# -999 VALUES 
def remove_999_features(xIn,fraction999Threshold):
    '''counts number of -999 values per feature and removes the ones with bigger percentage 
    than the fraction999Threshold '''
    x =np.copy(xIn)
    numFeatures=len(x[0,:])
    numTrials=len(x[:,0])
    x[x == -999] = np.nan
    num_999 = []
    frac_999 = []
    for c in range(0, numFeatures):
        cnt=np.count_nonzero(np.isnan(x[:, c]))
        num_999.append(cnt)
        frac_999.append(np.divide(cnt,numTrials))
    frac_999=np.asarray(frac_999)
    featuresToKeepIndx=np.where(frac_999<fraction999Threshold)
    y=xIn[:,featuresToKeepIndx]
    y=np.squeeze(y)
    print('Number of features after removal of the ones with too much 999:', len(y[0,:]))
    return y
    

def resolve999values(x):
    '''counts number of -999 values per feature and reports some statistics
    removes -999 values and replaces them with the mean of that feature
    constructs new features which are cathegorical features stating when some of the features had -999 values
    these newfeatures can be latter appended to the whole dataset as additional features'''
    num_features= len(x[0,:])
    num_samples = len(x[:,0])
    # counting number of them 
    indx999all=[]
    for i in range(num_features):
        true999= (x[:,i]==(-999))
        indx999=np.where(true999)
        indx999all=np.append(indx999all,indx999)   
    indx999set=set(indx999all)
    print('Number of samples when at leas one feature has value -999:', len(indx999set))
    print('Percentage of total data: ', len(indx999set)/num_samples)
        
    # swaping -999 with mean value of that feature 
    # and creating a new features if chosen 
    newFeatures=np.zeros([num_samples,1])
    for i in range(num_features):
        true999=(x[:,i]==(-999))
        false999=(x[:,i]!=(-999))
        indxTrue999=np.where(true999)
        indxFalse999=np.where(false999)
        meanVal=np.mean(x[indxFalse999,i])
        x[indxTrue999,i]=meanVal
        #creating new features 
        newFeature=1*true999
        if(np.sum(newFeature)!=0):
            newFeatures=np.column_stack((newFeatures,newFeature))
    newFeatures=newFeatures[:,1:]
    return x,newFeatures

# REMOVING OUTLIERS
def removeOutliers(x, numSigmas):
    ''' remove outliers from each feature according to the boundary set 
    boundary is expressed in number of standart deviations from mean 
    outliers are set to boundary values 
    also returns difference from outlier value and border value just for inspection if necessary ''' 
    xorg=np.copy(x)
    num_features= len(x[0,:])
    means=np.mean(x,axis=0)
    stds=np.std(x,axis=0)
    bordLow=means-numSigmas*stds
    bordUp=means+numSigmas*stds
    for i in range (num_features):
        x[:,i]=np.clip(x[:,i],bordLow[i],bordUp[i])
    return x, xorg-x

# BUCKETING
def bucket_features(x,numBuckets):
    ''' creates "numBuckets" of new additional cathegorical features for each original feature 
    so that they select parts of original features
    this should help to present each feature as linear combination of its parts 
    here in every bucket is the same RANGE OF VALUES from each features''' 
    numOriginalFeatures = len(x[0, :])
    newFeatures = np.zeros([len(x[:, 0]), (numBuckets * numOriginalFeatures)])
    indx=0
    minValuesFeatures=np.amin(x,axis=0)
    maxValuesFeatures = np.amax(x,axis=0)
    deltas=(maxValuesFeatures-minValuesFeatures)/numBuckets
    for i in range(numOriginalFeatures):
        for j in range(1,numBuckets):
            border=minValuesFeatures[i]+j*deltas[i]
            newFeatures[:, indx] = 1* (x[:,i]< border)
            indx=indx+1
    return newFeatures

def bucket_features2(x,numBuckets):
    ''' creates "numBuckets" of new additional cathegorical features for each original feature 
    so that they select parts of original features
    this should help to present each feature as linear combination of its parts 
    here in every bucket is the same NUMBER OF POINTS s from each features'''
    numOriginalFeatures = len(x[0, :])
    numSamples=len(x[:,0])
    newFeatures = np.zeros([len(x[:, 0]), (numBuckets * numOriginalFeatures)])
    indx=0
    deltas=int(numSamples/numBuckets)
    for i in range(numOriginalFeatures):
        sortedIndxs=np.argsort(x[:,i])
        for j in range(1,numBuckets+1):
            if (j==numBuckets):
                bucketIndxs=sortedIndxs[(j-1)*deltas:]
            else: 
                bucketIndxs=sortedIndxs[(j-1)*deltas:j*deltas]
            nf=np.zeros([numSamples])
            nf[bucketIndxs]=1
            newFeatures[:, indx] = nf
            indx=indx+1
    return newFeatures

#ETA THETA TRANSFORMS
def eta_to_theta(x):
    """converts particle direction from detector x-y plane to theta (spherical coordinates) on one column"""
    theta=2*np.arctan(np.exp(-x))
    return theta

def eta_to_theta_multiple_and_append(totransform, X):
    """converts particle direction from ATLAS x-y plane to theta (spherical coordinates) for multiple columns
    and appends them to dataset
    totransform: array of columns to transform
    X: dataset to which the columns are to be appended
    returns original X and appended new transformed columns """
    for i in range(len(totransform[0,:])):
        tmp=eta_to_theta(totransform[:,i])
        X=np.column_stack((X,tmp))
    return X


#TOTAL FEATURE PREPROCESING PIPELINE 
def featuresPreprocessing(x,remove999FeaturesOn, fraction999Threshold, createNew999Features,createNewQuadraticFeatures,createNewDegreeFeatures,degree, addSinCosFeatures, toGaussDistributionOn,skewnessThreshold,removeOutliersOn,numSigmas,bucketing0n,numBucketsPerFeature):
    print('Initial size of features:', np.shape(x))
    # if there is too much 999 in some feature remove it
    if (remove999FeaturesOn==1):
        x=remove_999_features(x,fraction999Threshold)
        print('Size after removel of too much 999 features:', np.shape(x))
    # put mean instead of 999 values
    x, new999Features=resolve999values(x)
    print('Size of new 999 features:', np.shape(new999Features))
    # removing outliers
    if (removeOutliersOn==1):
        x=removeOutliers(x, numSigmas)
    # correct distribution to gaussian
    if (toGaussDistributionOn==1):
        x=transform_to_gauss(x,skewnessThreshold)
    # building polinomial features
    xExp=build_poly(x,createNewDegreeFeatures,createNewQuadraticFeatures, degree)
    print('Size after expanding with polinoms:', np.shape(xExp))
    # adding sin and cos features
    if (addSinCosFeatures==1):
       newFeatures=build_sincos_features(x) #from non expanded features
       xExp = np.column_stack((xExp, newFeatures))
       print('Size after adding sincos features:', np.shape(xExp))
    # add new features where there were 999s in original features (they don't have to be expanded with poly, sincos etc.
    if (createNew999Features==1):
        xExp = np.column_stack((xExp, new999Features))
    print('Final size:', np.shape(xExp))
    #normalizing data
    #x_norm, mean_x, std_x= standardize01(xExp)
    x_norm, mean_x, std_x= standardize(xExp)
    # bucketing original features
    if (bucketing0n == 1):
        newFeatures=bucket_features(x, numBucketsPerFeature)
        x_norm = np.column_stack((x_norm, newFeatures))
        print('Size after adding bucketing features:', np.shape(x_norm))
    # addind 1 for firs column
    x_out=augument_feature(x_norm)
    return x_out


def featuresPreprocessing2(x,params):  
    print('Initial size of features:', np.shape(x))
    # if there is too much 999 in some feature remove it
    if (params["remove999FeaturesOn"]==1):
        x=remove_999_features(x,params["fraction999Threshold"])
        print('Size after removel of too much 999 features:', np.shape(x))
    # put mean instead of 999 values
    x, new999Features=resolve999values(x)
    print('Size of new 999 features:', np.shape(new999Features))
    # removing outliers
    if (params["removeOutliersOn"]==1):
        x,outliers=removeOutliers(x, params["numSigmas"])
    # correct distribution to gaussian
    if (params["toGaussDistributionOn"]==1):
        x=transform_to_gauss(x,params["skewnessThreshold"])
    # building polinomial features
    xExp=build_poly(x,params["createNewDegreeFeatures"],params["createNewQuadraticFeatures"], params["degree"])
    print('Size after expanding with polinoms:', np.shape(xExp))
    # adding sin and cos features
    if (params["addSinCosFeatures"]==1):
       newFeatures=build_sincos_features(x) #from non expanded features
       xExp = np.column_stack((xExp, newFeatures))
       print('Size after adding sincos features:', np.shape(xExp))
    #adding eta theta tranform features
    if (params["addEtaThetaFeatures"]==1):
        xExp=eta_to_theta_multiple_and_append(x, xExp)
     # correct distribution to gaussian
#    if (params["toGaussDistributionOn"]==1):
#        xExp=transform_to_gauss(xExp,params["skewnessThreshold"])
#    # add new features where there were 999s in original features (they don't have to be expanded with poly, sincos etc.
    if (params["createNew999Features"]==1):
        xExp = np.column_stack((xExp, new999Features))
    print('Final size:', np.shape(xExp))
    #normalizing data
    #x_norm, mean_x, std_x= standardize01(xExp)
    x_norm, mean_x, std_x= standardize(xExp)
    # bucketing original features
    if (params["bucketing0n"] == 1):
        newFeatures=bucket_features2(x, params["numBucketsPerFeature"])
        x_norm = np.column_stack((x_norm, newFeatures))
        print('Size after adding bucketing features:', np.shape(x_norm))
   # addind 1 for firs column
    x_out=augument_feature(x_norm)
    return x_out
