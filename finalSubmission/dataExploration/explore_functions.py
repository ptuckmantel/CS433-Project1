import numpy as np
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt

def replace_nans(x, y):
    """Separates training data into signal and background classes and replaces missing data"""
    "with the mean of each class"
    
    N = x.shape[0]
    D = x.shape[1]
    
    x_ones = x[y == 1., :]
    x_zeros = x[y == 0., :]

    x_ones_means = np.nanmean(x_ones, axis=0)
    x_zeros_means = np.nanmean(x_zeros, axis=0)

    for i in range(N):
        nans_in_x = np.argwhere(np.isnan(x[i, :]))
        if y[i] == 1:
            x[i, nans_in_x] = x_ones_means[nans_in_x]
        if y[i] == 0.:
            x[i, nans_in_x] = x_zeros_means[nans_in_x]
            
    return x, x_ones, x_zeros


def replace_nans_gen_mean(x):
    """Replaces missing data with mean of each feature"""

    N = x.shape[0]

    x_clean_means = x

    x_means = np.nanmean(x, axis=0)

    for i in range(N):
        nans_in_x = np.argwhere(np.isnan(x[i, :]))
        x_clean_means[i, nans_in_x] = x_means[nans_in_x]

    return x_clean_means, x_means


def zscore_std(x):
    x_mean = np.nanmean(x, axis=0)
    x_std = np.nanstd(x, axis=0)
    
    x_z = np.divide((x - x_mean), x_std)
    
    return x_z


def norm_x(x):
    x_norm = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    
    return x_norm

def cube_transform(x):
    cubed_x = x**3
    
    return cubed_x

def log_transform(x):
    shifted_x = x - np.min(x) + 1
    
    return np.log(shifted_x)
    
def transform_to_gauss(x):
    corrected_data=[]
    D = x.shape[1]
    
    for d in range(D):
        skewness = skew(x[:, d])
        if skewness < -0.5:
            corrected_data.append(cube_transform(x[:, d]))
        elif skewness > 0.5:
            corrected_data.append(log_transform(x[:, d]))
        else:
            corrected_data.append(x[:, d])
            
    return np.array(corrected_data).T


def plot_scatter_matrix(dataframe):
    """Does a scatter matrix plot of input dataframe"""
    scatter_matrix = pd.plotting.scatter_matrix(dataframe,diagonal = "kde", figsize=(20,20))
    for ax in scatter_matrix.ravel():       
        ax.set_xlabel(ax.get_xlabel(), fontsize = 10, rotation = 90,)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 10, rotation = 0, labelpad=60)

        
def plot_radviz(dataframe, target):
    """Does a radviz plot of input dataframe using the target variable as outcome. Outcomes are color coded in green and blue"""
    plt.figure(figsize=(24,18))
    pd.plotting.radviz(dataframe, target, color=('green', 'blue'), alpha=0.5, marker='.')

    
def plot_parallel_coord(dataframe, target):
    """Does a parallel coordinate plot of input dataframe using target variable as outcome. Outcomes are color coded in green and blue"""
    plt.figure(figsize=(24,18))
    pd.plotting.parallel_coordinates(dataframe, target, color=('green','blue'))
    plt.xticks(rotation=90)

    
def remove_invalid_data(data, invalid_value):
    """ Removes all rows where invalid_value is present"""
    data=data[~(data==invalid_value).any(axis=1)]
    
    return data
    
     















