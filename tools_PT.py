from scipy.stats import skew


def cube_transform(x):
    """returns the 3rd power of all elements in input array"""
    cubed_x = x ** 3
    return cubed_x


def log_transform(x):
    """returns the log transform of all elements in input array. Values are first shifted to all positives"""
    shifted_x = x - np.min(x) + 1
    return np.log(shifted_x)


def skewness_correction(x):
    """ Corrects the data for each column by applying cube transform if skewness is smaller than -0.5 and log if larger than +0.5"""
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