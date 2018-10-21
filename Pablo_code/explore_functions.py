import numpy as np
from scipy.stats import skew

def replace_nans(x, y):
    "Separates training data into signal and background classes and replaces missing data"
    "with the mean of each class"
    
    N = x.shape[0]
    D = x.shape[1]
    
    x_ones = x[y == 1., :]
    x_zeros = x[y == -1., :]

    x_ones_means = np.nanmean(x_ones, axis=0)
    x_zeros_means = np.nanmean(x_ones, axis=0)

    for i in range(N):
        nans_in_x = np.argwhere(np.isnan(x[i, :]))
        if y[i] == 1:
            x[i, nans_in_x] = x_ones_means[nans_in_x]
        if y[i] == -1:
            x[i, nans_in_x] = x_zeros_means[nans_in_x]
            
    return x, x_ones, x_zeros

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
     














