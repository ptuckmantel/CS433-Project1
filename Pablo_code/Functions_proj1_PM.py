# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def compute_mse(e):
    e_mse = 0.5*np.mean(e**2)
    return e_mse 
def compute_mae(e):
    e_mae = np.mean(np.abs(e))

def compute_loss(y, tx, w, mse):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    #print(e)
    if mse:
        return compute_mse(e)
    else:
        return compute_mae(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""

    err = y - tx.dot(w)
    loss = compute_loss(y, tx, w, 1)
    gradient = -tx.T.dot(err) / len(err)
    return err, loss, gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        err, loss, gradient = compute_gradient(y, tx, w)
      
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws