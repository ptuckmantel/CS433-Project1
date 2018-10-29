from scipy.stats import skew
import numpy as np


def cube_transform(x):
    """returns the 3rd power of all elements in input array"""
    cubed_x = x ** 3
    return cubed_x


def log_transform(x):
    """returns the log transform of all elements in input array. Values are first shifted to all positives"""
    shifted_x = x - np.min(x) + 1
    return np.log(shifted_x)


def skewness_correction(x):
    """ Reduces skewness for each column by applying cube transform if skewness is <-0.5 and log if >+0.5"""
    corrected_data = []
    num_cols = np.shape(x)[1]
    for i in range(num_cols):
        this_col = x[:, i]
        skewness = skew(this_col)
        if skewness < -0.5:
            corrected_data.append(cube_transform(this_col))
        elif skewness > 0.5:
            corrected_data.append(log_transform(this_col))
        else:
            corrected_data.append(this_col)
    return np.array(corrected_data).T


def eta_to_theta(x):
    """converts particle direction from detector x-y plane to theta (spherical coordinates) on one column."""
    theta=2*np.arctan(np.exp(-x))
    return theta

def eta_to_theta_multiple_and_append(totransform, X):
    """Converts particle direction from ATLAS x-y plane to theta (spherical coordinates) for multiple columns and appends them to dataset.
    Input:
    totransform: array of columns to transform
    X: dataset to which the columns are to be appended

    Returns: array consisting columns of X and transformed columns
    """
    totransform=totransform.T
    newcols=[]
    for i in range(totransform.shape[1]):
        tmp=eta_to_theta(totransform[:,i])
        X=np.column_stack((X,tmp))
    return X