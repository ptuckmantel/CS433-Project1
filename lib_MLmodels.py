# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:09:16 2018

@author: Una
"""
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
from lib_dataPreprocessing import *
from lib_errorMetrics import *

# COST FUNCTIONS 

def loss_mse(y, tx, w):
    e = y - tx.dot(w)
    loss=1/2*np.mean(e**2)
    return loss

def loss_mae(y, tx, w):
    e = y - tx.dot(w)
    loss=np.mean(np.abs(e))
    return loss

def loss_rmse(y, tx, w):
    e = y - tx.dot(w)
    loss=np.sqrt(np.mean(e**2))
    return loss



# LEAST SQUARES REGRESSION 

def least_squares(y,tx):
    #Finding optimal weights
    a=tx.T.dot(tx)
    #a=np.matmul(tx.transpose(),tx)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    
    loss=loss_rmse(y, tx, w)
    return w,loss


# GRADIENT DESCENT 

def compute_gradient(y, tx, w):
    N=len(y)
    e=y-tx.dot(w)
    grad=(-1*tx.transpose().dot(e))/N
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad=compute_gradient(y,tx,ws[n_iter]);
        loss=loss_rmse(y, tx, ws[n_iter]); #MSE loss 
        # update w by gradient
        w=ws[n_iter]-gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad=compute_gradient(y,tx,ws[n_iter]);
            loss=loss_rmse(y, tx, ws[n_iter]); # MSE loss
            # update w by gradient
            w=ws[n_iter]-gamma*grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
            
    return losses, ws
    
def gradient_descent(yb,features,max_iters,gamma,plottingOn):
    w_initial = np.zeros(np.shape(features[1,:]))
#    max_iters =500
#    gamma =0.05 # 0.7
    
    RMSE_gd, w_gd= least_squares_GD(yb, features, w_initial, max_iters, gamma)
    iterations=np.arange(0, max_iters, 1)
    w_gd_arrray=np.asarray(w_gd)
    RMSE_gd_arrray=np.asarray(RMSE_gd)
    w=w_gd_arrray[max_iters,:]
    
    #plot reducing RMSE through iterations
    if (plottingOn==1):
        fig=plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iterations, RMSE_gd) #marker='o', color='w', markersize=10
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid()
    return w,RMSE_gd_arrray[max_iters-1]


# RIDGE REGRESSION 

def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    #Finding optimal weights
    D=len(tx[0,:])
    N=len(y)
    a=tx.T.dot(tx)+2*lambda_*N*np.identity(D)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)

    loss=loss_mse(y, tx, w)  # MSE loss or RMSE ???
    return w, loss

def optimal_ridge_regression(yb,tx, lambdas,plottingOn):
    losses=[]
    weights=np.zeros([len(tx[0,:]),len(lambdas)])
    for i in range(len(lambdas)):
        w,loss=ridge_regression(yb, tx, lambdas[i])
        losses=np.append(losses,loss)
        weights[:,i]=w
    lambda_indx=np.argmin(losses)
    lambdaDecision=lambdas[lambda_indx]
    w=weights[:,lambda_indx]
    if (plottingOn==1):
        #iterations=np.arange(0, len(lambdas), 1)
        fig=plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(lambdas, losses) #marker='o', color='w', markersize=10
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid()
    return w, losses[lambda_indx],lambdaDecision

# LOGISTIC REGRESSION 
    
def sigmoid(t):
    """apply sigmoid function on t."""
    sig=np.exp(t)/(1+np.exp(t))
#    print('...t:',t)
    #sig=1.0 / (1 + np.exp(-t))
    return sig

def loss_maxL(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    sig=sigmoid(tx.dot(w))
    p=y.T.dot(np.log(sig)) + (1-y).T.dot(np.log(1-sig))
    p=p/len(y)
#    print('txw:', tx.dot(w))
#    print('...sig:',sig)
#    print('...log(sig):', np.log(sig))
#    print('...log(1-sig):', np.log(1-sig))
#    print('...f:',y.T.dot(np.log(sig)))
#    print('...s:',(1-y).T.dot(np.log(1-sig)) )
#    print('...p:', p)
    #print(np.shape(p))
    return np.squeeze(- p)

def calculate_maxL_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig=sigmoid(tx.dot(w))
    grad=tx.T.dot(sig-y)
    grad=grad/len(y)
#    print('***tx:', tx)
#    print('***sig-y:', sig-y)
#    print('***grad:',grad)
    
    return grad 

def step_maxL_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    
    loss=loss_maxL(y, tx, w)
    grad=calculate_maxL_gradient(y,tx,w)
#    print()
#    print()
#    print('---w:',w)
#    print('--- grad:', grad, ' loss:', loss)
    
    # update w by gradient
    w=w-gamma*grad
    return w, loss

def step_maxL_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    grad=calculate_maxL_gradient(y,tx,w)+ 2*lambda_*w # shuld here be 2 ????
    loss=loss_maxL(y, tx, w)+ lambda_* np.linalg.norm(w)**2
    # update w by gradient
    w=w-gamma*grad
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    threshold=0.000001
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        w, loss= step_maxL_gradient_descent(y, tx, ws[n_iter], gamma)
        # store w and loss
        ws.append(w)
        losses.append(loss)
#        if len(losses) > 3:
#            print('--loss:', loss)
#            print('--loss-1:', losses[-1])
#            print('--loss-2:', losses[-2])
#            print('---:', np.abs(losses[-1] - losses[-2]))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
            #print("Current iteration={i}, loss={l},  w={we}".format(i=n_iter, l=loss, we=w))
    #return ws, losses
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    threshold=1
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        w, loss= step_maxL_penalized_gradient_descent(y, tx, ws[n_iter], gamma, lambda_)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print('loss-1:',losses[-1])
            print('loss-2:',losses[-2])
            print('##############')
            break
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    #return ws, losses
    return w, loss

# CROSS VALIDATION 

def build_k_indices(num_row, k_fold, seed):
    """build k indices for k-fold
    returns array/matrix of indices separated into k folds (rows)"""
    #num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)