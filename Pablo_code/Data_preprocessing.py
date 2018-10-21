# Preprocessing of CERN's Higgs boson challenge data, for project 1
# Pablo Maceira
# 13.10.2018


#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] =(24,18)
import pandas as pd
from pandas.plotting import lag_plot, scatter_matrix, radviz
import seaborn as sns
from proj1_helpers import *
from Functions_project1_PM import *
from os import path
from scipy import stats

dir_path = path.dirname(path.realpath("Data_preprocessing.py"))
data_path = path.abspath(path.join(dir_path, "../all/",  "train.csv"))
# print(filepath)

yb, input_data, ids = load_csv_data(data_path, sub_sample=False)
print("yb has size ", yb.shape)
print("input data x has size ", input_data.shape)
print("The ids have size ", ids.shape)


x = input_data
y = yb

df_x = pd.DataFrame(x)
df_y = pd.DataFrame({'outcome': y})
data = [df_y, df_x]
df = pd.concat(data, axis=1)
df_sample = df.sample(1000)

# Data exploration
# Scatter matrix
# scatter_matrix(df, alpha=0.2, diagonal='kde')
# Lag plot
#lag_plot(df, alpha=0.2)
# RadViz
#radviz(df, 'Name', )


max_iters = 1000
gamma = 1/max_iters
N = x.shape[0]
D = x.shape[1]
initial_w = np.zeros(D)

x = x
x[x == -999] = np.nan
num_nas = []
nas_frac = []
for c in range(0, 30):
    num_nas.append(np.count_nonzero(~np.isnan(x[:, c])))
    nas_frac.append(np.divide((x.shape[0] - num_nas[c]), x.shape[0]))


# Correct for missing data. I will use the average of every feature within the same label
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

t_test = stats.ttest_ind(x_ones[:, 0], x_zeros[:, 0], equal_var=False, nan_policy='omit')

# Plot data to see distributions
# fig = plt.figure(1)
# fig.suptitle("All features")
# for feat_idx in range(1, D):
#    plt.subplot(6, 5, feat_idx)
#    plt.scatter(y, x[:, feat_idx])

# loss_LS, w_LS = least_squares(y, x)
# print('Loss was ', loss_LS)
# print('Estimated parameters are ', w_LS)

# loss_end, w_end = gradient_descent(y, x, initial_w, max_iters, gamma)



