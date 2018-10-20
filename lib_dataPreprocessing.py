# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:45:24 2018

@author: Una
"""
# PARAMETERS THAT ARE NEEDED AT SOME POINT FOR THIS FUNCTIONS 
# best to specify them at the beginning of the main scrip 
#plottingOn=0
#createNew999Features=0
#createNewDegreeFeatures=0
#createNewQuadraticFeatures=1
#degree=2


# DATA NORMALIZATION 
def standardize01(x):
    """Standardize the original data set to values between 0 and 1 ."""
    min_x = np.amin(x, axis=0)
    max_x = np.amax(x, axis=0)
    x_norm = (x - min_x)/(max_x-min_x)
    std_x = np.std(x_norm,axis=0)
    mean_x = np.mean(x_norm,axis=0)
    return x_norm, mean_x, std_x

def standardize(x):
    """Standardize the original data set so that mean= and std=1."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x


# BUILD POLINOMIAL BASE
    
def build_poly_degree(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree
    - do the same for eaxh feature, but dont combine them ."""
    numOriginalFeatures=len(x[0,:])
    newFeatures=np.zeros([len(x[:,0]),((degree-1)*numOriginalFeatures)])
    indx=0
    for i in range(numOriginalFeatures):
        for j in range(2,degree+1):
            newFeatures[:,indx]=np.power(x[:,i],j)
            indx=indx+1
    return newFeatures

def build_quadratic_poly(x):
    """polynomial basis functions with degree=2 but with all possible
    recombinations of features """
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
    print('---')
    print('Features dimension before polynomial basis:', np.shape(x))
    if (createNewDegreeFeatures==1):
        newFeatures1=build_poly_degree(x, degree)
    if (createNewQuadraticFeatures==1):
        newFeatures2=build_quadratic_poly(x)
    if (createNewDegreeFeatures==1):
        x=np.column_stack((x,newFeatures1))
    if (createNewQuadraticFeatures==1):
        x=np.column_stack((x,newFeatures2))
    print('Features dimension after polynomial basis:', np.shape(x))
    print('---')
    return x


# AUTUMENTING FEATURES
def augument_feature(x):
    """add column of ones at the beginning """
    col1=np.ones([len(x[:,0]),1])
    x=np.column_stack((col1,x))
    return x


# -999 VALUES 
def resolve999values(x,createNew999Features,plottingOn):
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
    if (createNew999Features==1):
        newFeatures=newFeatures[:,1:]
        x=np.column_stack((x,newFeatures))
        print('Features dimension after 999 expansion:', np.shape(x))    
        
    # plotting expanded features 
    num_features2=len(x[0,:])
    if (plottingOn==1):
        fig=plt.figure(figsize=(50, 25), dpi= 40)
        for i in range(num_features2):
            ax1 = fig.add_subplot(12, 5, i+1)
            ax1.scatter(samples, xExp[:,i],marker='.');
            ax1.set(xlabel='sample', ylabel='feature value',title='Feature '+ str(i+1))
            ax1.grid()
        fig.savefig("expanded_features")
    return x

# REMOVING OUTLIERS
def removeOutliers(x, percentileOutliers):
    num_features= len(x[0,:])
    x_perc_border=np.percentile(x, percentileOutliers, axis=0)
    for i in range (num_features):
        x[:,i]=np.clip(x[:,i],0,x_perc_border[i])
    return x