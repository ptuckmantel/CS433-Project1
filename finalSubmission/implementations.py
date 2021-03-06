# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:09:16 2018

@author: Una
"""
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *

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
    ''' least squares algorithm'''
    #finding optimal weights
    a=tx.T.dot(tx)
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
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING> """
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
    """gradient descent algorithm"""
    # fefine parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad=compute_gradient(y,tx,ws[n_iter]);
        loss=loss_rmse(y, tx, ws[n_iter]); #MSE loss 
        loss=loss/len(y)
        grad=grad
        # update w by gradient
        w=ws[n_iter]-gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):  
    """stochastic gradient descent algorithm"""
    # define parameters to store w and loss
    batch_size=1
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
    return loss, w


# RIDGE REGRESSION 

def ridge_regression(y, tx, lambda_):
    """ridge regression"""
    #finding optimal weights
    D=len(tx[0,:])
    N=len(y)
    a=tx.T.dot(tx)+2*lambda_*N*np.identity(D)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    loss=loss_rmse(y, tx, w)
    return w, loss

def optimal_ridge_regression(yb,tx, lambdas,plottingOn):
    ''' performs ridge for different values of lambda ans chooses the one with minimal loss
    possibility to plot how is loss dependant on lambda ''' 
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
        fig=plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(lambdas, losses) #marker='o', color='w', markersize=10
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid()
    return w, losses[lambda_indx],lambdaDecision

# LOGISTIC REGRESSION 
    
def sigmoid(t):
    """apply sigmoid function on t"""
    sig=np.exp(t)/(1+np.exp(t))
    return sig

def loss_maxL(y, tx, w):
    """compute the cost by negative log likelihood."""
    # to avoid problems when 0 in log
    epsilon=0.00000001
    sig=sigmoid(tx.dot(w))
    # calculated probability
    p=y.T.dot(np.log(sig+epsilon)) + (1-y).T.dot(np.log(1-sig+epsilon))
    #divides with number of samples so that learning rate is not dependant on number of samples
    p=p/len(y)
    return np.squeeze(- p)

def calculate_maxL_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig=sigmoid(tx.dot(w))
    grad=tx.T.dot(sig-y)
    #divides with number of samples so that learning rate is not dependant on number of samples
    grad=grad/len(y) 
    return grad 

def step_maxL_gradient_descent(y, tx, w, gamma):
    """does one step of gradient descent using logistic regression
    returns the loss and the updated w """
    loss=loss_maxL(y, tx, w)
    grad=calculate_maxL_gradient(y,tx,w)
    # update w by gradient
    w=w-gamma*grad
    return w, loss

def step_maxL_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """ does one step of gradient descent, using the penalized logistic regression
    returns the loss and updated w"""
    grad=calculate_maxL_gradient(y,tx,w)+ 2*lambda_*w 
    loss=loss_maxL(y, tx, w)+ lambda_* np.linalg.norm(w)**2
    # update w by gradient
    w=w-gamma*grad
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ''' logistic regression algorithm'''
    # threshold to stop if increase in loss is too small
    threshold=0.00000001
    # define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        w, loss= step_maxL_gradient_descent(y, tx, ws[n_iter], gamma)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ''' regularized logistic regression algorithm'''
    # threshold to stop if increase in loss is too small
    threshold=0.00000001
    # define parameters to store w and loss
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
            break
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    return w, loss
