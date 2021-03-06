import numpy as np

""" BASIC DATA PREPROCESSING AND FEATURE TRANSFORMATION"""


def standardise(a):
    """Centers data around mean and normalised by standard deviation. Returns standardised data, its mean and standard deviation"""
    mean = np.mean(a, axis=0)
    stddev = np.std(a, axis=0)
    b = (a - mean) / stddev
    return b, mean, stddev


def augment_data(x):
    """ adds a column of ones to the input data"""
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    return tx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        temp = np.power(x, d)
        poly = np.c_[poly, temp]
    return poly


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    n = len(y)
    tr_length = np.int(n * ratio)
    all_indices=np.arange(len(y))
    tr_indices = np.random.choice(np.arange(n), tr_length, replace=False)
    te_indices = np.array([i for i in all_indices if i not in tr_indices])
    #x_train = x[indices]
    #y_train = y[indices]
    x_train = x[tr_indices]
    y_train = y[tr_indices]
    x_test=x[te_indices]
    y_test = y[te_indices]
    #x_test = np.array([i for i in x if i not in x_train])
    #y_test = np.array([i for i in y if i not in y_train])

    return x_train, y_train, x_test, y_test


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold. Returns a k-fold long array containing each fold's indices"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    all_indices = np.arange(len(y))
    test_indices = k_indices[k]
    train_indices = [i for i in all_indices if i not in test_indices]
    xtest, ytest = x[test_indices], y[test_indices]
    xtrain, ytrain = x[train_indices], y[train_indices]
    # form data with polynomial degree
    xtrain_poly = build_poly(xtrain, degree)
    xtest_poly = build_poly(xtest, degree)
    # ridge regression
    w_opt = ridge_regression(ytrain, xtrain_poly, lambda_)
    # w_opt=ridge_regression(ytrain, xtrain_poly, lambda_)
    # calculate the loss for train and test data
    loss_tr = compute_rmse(ytrain, xtrain_poly, w_opt)
    loss_te = compute_rmse(ytest, xtest_poly, w_opt)

    return loss_tr, loss_te, w_opt


""" COST FUNCTIONS"""


def compute_e(y, tx, w):
    """compute the error vector"""
    e = y - np.dot(tx, w)
    return e


def compute_loss_mse(y, tx, w):
    """compute mean squared error"""
    diff = compute_e(y, tx, w)
    loss = 0.5 * np.mean(diff ** 2)
    return loss


def compute_loss_mae(y, tx, w):
    """compute mean absolute error"""
    diff = compute_e(y, tx, w)
    loss = 0.5 * np.mean(np.abs(diff))
    return loss


def compute_rmse(y, tx, w):
    """compute root mean squared error"""
    mse = compute_loss_mse(y, tx, w)
    rmse = np.sqrt(2 * mse)
    return rmse


# def compute_loss_logistic(y, tx, w):
#     """compute the cost by negative log likelihood."""
#     y_hat = np.dot(tx, w)
#     loss_vector = np.log(1 + np.exp(y_hat)) - y * y_hat
#     loss = np.sum(loss_vector)
#     loss=loss/len(y)
#     return loss
#
def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    epsilon=1e-8
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred+epsilon)) + (1 - y).T.dot(np.log(1 - pred+epsilon))
    loss=loss/len(y)
    return np.squeeze(- loss)



"""TOOLS FOR COST OPTIMISATION"""


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


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def compute_gradient_mse(y, tx, w):
    """Compute the gradient using MSE."""
    e = compute_e(y, tx, w)
    n = len(y)
    gradient = -np.dot(tx.T, e) / n
    return gradient


def compute_gradient_mae(y, tx, w):
    """Compute the gradient using MAE"""
    e = compute_e(y, tx, w)
    gradient = np.linalg.norm(e)
    return gradient


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples and their corresponding labels."""
    e = compute_e(y, tx, w)
    n = len(y)
    gradient = -np.dot(tx.T, e) / n
    return gradient

def sigmoid(t):
    """apply sigmoid function on t."""
    #s = 1. / (1 + np.exp(-t))
    t=np.clip(t, -500, 500)
    s=np.exp(t)/(1+np.exp(t))
    return s

#
# def compute_gradient_logistic(y, tx, w):
#     """compute the gradient of loss."""
#     y_hat = np.dot(tx, w)
#     s = sigmoid(y_hat)
#     grad = np.dot(tx.T, (s - y))/len(y)
#     return grad

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of logistic cost function."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)/len(y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_logistic(y, tx, w)
    grad = compute_gradient_logistic(y, tx, w)
    w = w - gamma * grad
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, and gradient of logistic cost function"""
    loss = compute_loss_logistic(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return w, loss


""" COST OPTIMISATION"""


def gridsearch(y, tx, w0, w1, loss_function='mse'):
    losses = np.zeros((len(w0), len(w1)))
    for row_ind, row in enumerate(w0):
        for col_ind, col in enumerate(w1):
            w = np.array([row, col])
            if loss_function == 'mse':
                losses[row_ind, col_ind] = compute_loss_mse(y, tx, w)
            elif loss_function == 'mae':
                losses[row_ind, col_ind] = compute_loss_mae(y, tx, w)
            else:
                print('invalid loss function')
    return losses


def gradient_descent(y, tx, initial_w, max_iters, gamma, status=True, cost_function='mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        if cost_function == "mse":
            loss = compute_loss_mse(y, tx, w)
            gradient = compute_gradient_mse(y, tx, w)
        elif cost_function == "mae":
            loss = compute_loss_mae(y, tx, w)
            gradient = compute_gradient_mae(y, tx, w)

        w = w - gamma * gradient

        ws.append(w)
        losses.append(loss)
        if status:
            print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1,
                                                                                    ls=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma, status=True):
    """Gradient descent algorithm with MSE cost function. Returns w, loss: the optimal weights and the loss"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        if status:
            print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1,
                                                                                    ls=loss, w0=w[0], w1=w[1]))
    return w, loss


def gradient_descent_dynamic(y, tx, initial_w, max_iters, status=True, cost_function='mse'):
    """Gradient descent algorithm with gamma=1/step_number."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        if cost_function == "mse":
            loss = compute_loss_mse(y, tx, w)
            gradient = compute_gradient_mse(y, tx, w)
        elif cost_function == "mae":
            loss = compute_loss_mae(y, tx, w)
            gradient = compute_gradient_mae(y, tx, w)
        gamma = 1 / (n_iter + 1)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if status:
            print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}, g={g}".format(
                bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1], g=gamma))
    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, status='True', cost_function='mse'):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    ws = [initial_w]
    losses = []
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            if cost_function == "mse":
                gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
                loss = compute_loss_mse(y, tx, w)
            elif cost_function == "mae":
                gradient = compute_gradient_mae(minibatch_y, minibatch_tx, w)
                loss = compute_loss_mae(y, tx, w)
            w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if status:
            print("S.G.D({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, status='True'):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    ws = [initial_w]
    losses = []
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            loss = compute_loss_mse(y, tx, w)
            w = w - gamma * gradient
        if status:
            print("S.G.D({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return w, loss



def stochastic_gradient_descent_dynamic(y, tx, initial_w, batch_size, max_iters, status='True', cost_function="mse"):
    """Stochastic gradient descent algorithm with gamma=1/step_number."""
    w = initial_w
    ws = [initial_w]
    losses = []
    for n_iter in range(max_iters):
        gamma = 1 / (n_iter + 1)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            if cost_function == "mse":
                gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
                loss = compute_loss_mse(y, tx, w)
            elif cost_function == "mae":
                gradient = compute_gradient_mae(minibatch_y, minibatch_tx, w)
                loss = compute_loss_mae(y, tx, w)
            w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if status:
            print("S.G.D({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return w, loss

def stochastic_reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma, batch_size):
    w=initial_w
    losses= []
    for n_iter in range(max_iter):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            _, grad = penalized_logistic_regression(minibatch_y, minibatch_tx, w, lambda_)
            loss=compute_loss_logistic(y, tx, w)
        w=w-gamma*grad
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """calculate the ridge regression solution"""
    identity = np.identity(np.shape(tx)[1])
    a = np.dot(tx.T, tx) + 2 * len(y) * lambda_ * identity
    b = np.linalg.inv(a)
    c = np.dot(b, tx.T)
    w = np.dot(c, y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """ Calculate w using logistic gradient descent"""
    loss_hist = []
    thresh=1e-5
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        loss_hist.append(loss)

        if len(loss_hist)>1 and np.abs(loss_hist[-1]-loss_hist[-2])<thresh:
            break
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    """Calculate w using regularised logistic regression"""
    thresh=1e-5
    loss_hist = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        loss_hist.append(loss)

        # log info
        if len(loss_hist)>1 and np.abs(loss_hist[-1]-loss_hist[-2])<thresh:
            break
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, loss


""" CHECKING MODEL """


def prediction(w0, w1, mean_x, std_x):
    """Get the regression line from the model."""
    x = np.arange(1.2, 2, 0.01)
    x_normalized = (x - mean_x) / std_x
    return x, w0 + w1 * x_normalized
