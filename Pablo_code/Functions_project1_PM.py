# -*- coding: utf-8 -*-
import numpy as np


def compute_mse(e):
    e_mse = 0.5*np.mean(e**2)
    return e_mse


def compute_mae(e):
    e_mae = np.mean(np.abs(e))
    return e_mae


def compute_loss(y, tx, w, mse):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)

    if mse:
        return e, compute_mse(e)
    else:
        return e, compute_mae(e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    e, loss = compute_loss(y, tx, w, 1)  # For MSE, argin(4)=1, for MAE, argin(4)=0
    gradient = -tx.T.dot(e) / len(e)
    return e, loss, gradient


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


def least_squares(y, tx):
    """calculate the least squares solution."""

    w_star = np.linalg.solve((tx.T.dot(tx)), tx.T.dot(y))
    err = (y - tx.dot(w_star)).T.dot((y - tx.dot(w_star)))
    # Means squared error cost function 1/(2N) * sum(y - x^Tw)^2
    n = len(tx)
    mse = (0.5 * err) / n
    # Optimal weights w^*. linalg.solve is the equivalent to Matlab's backslash

    # print(w_star)
    return mse, w_star
