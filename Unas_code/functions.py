import numpy as np 


# coding: utf-8

# ## Cost functions
# 

# In[ ]:


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


# ## Least squares regression 

# In[ ]:


def least_squares(y,tx):
    #Finding optimal weights
    a=tx.T.dot(tx)
    #a=np.matmul(tx.transpose(),tx)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    
    loss=loss_rmse(y, tx, w)
    return w,loss


# ## Gradient descent 

# In[8]:


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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            grad=compute_gradient(y,tx,ws[n_iter]);
            loss=loss_rmse(y, tx, ws[n_iter]); # MSE loss
            # update w by gradient
            w=ws[n_iter]-gamma*grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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

# ## Ridge regression
# 

# In[1]:


def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    #Finding optimal weights
    D=len(tx[0,:])
    N=len(y)
    a=tx.T.dot(tx)+2*lambda_*N*np.identity(D)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)

    loss=loss_mse(y, tx, w)  # MSE loss
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
        ax1.plot(lambdas, RMSE_rr) #marker='o', color='w', markersize=10
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid()
    return w, losses[lambda_indx],lambdaDecision
# ## Various 

# In[ ]:

def standardize01(x):
    """Standardize the original data set to values between 0 and 1 ."""
    min_x = np.amin(x, axis=0)
    max_x = np.amax(x, axis=0)
    x_norm = (x - min_x)/(max_x-min_x)
    std_x = np.std(x_norm,axis=0)
    mean_x = np.mean(x_norm,axis=0)
    return x_norm, mean_x, std_x

def standardize(x):
    """Standardize the original data set so that mean= and std=1."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x



#ERROR METRICS
# confusion matrix 
def conf_mat(y_pred, y_true):
        # confusion matrix
        conf_mat = [[0, 0], [0, 0]]

        for i in range(len(y_pred)):
                if int(y_true[i]) == 1: 
                        if (y_pred[i]==1):
                                conf_mat[0][0] = conf_mat[0][0] + 1 #TP
                        else:
                                conf_mat[1][0] = conf_mat[1][0] + 1 #FN
                elif int(y_true[i]) == (-1):
                        if (y_pred[i]==1):
                                conf_mat[0][1] = conf_mat[0][1] +1 #FP
                        else:
                                conf_mat[1][1] = conf_mat[1][1] +1 #TN

        accuracy = float(conf_mat[0][0] + conf_mat[1][1])/(len(y_true))
        precision = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[0][1])  
        sensitivity = float(conf_mat[0][0])/float(conf_mat[0][0] + conf_mat[1][0])
        specificity = float(conf_mat[1][1])/float(conf_mat[1][1] + conf_mat[1][0]) 
        return conf_mat, accuracy, precision, sensitivity, specificity 
    
    
# BUILD POLINOMIAL BASE
def build_poly_degree(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree
    - do the same for eaxh feature, but dont combine them ."""
    numOriginalFeatures=len(x[0,:])
    #out=np.matrix(np.ones([len(x),1]))
    newFeatures=np.zeros([len(x[:,0]),((degree-1)*numOriginalFeatures)])
    indx=0
    for i in range(numOriginalFeatures):
        for j in range(2,degree+1):
            newFeatures[:,indx]=np.power(x[:,i],j)
            indx=indx+1
    #out=np.column_stack((x,newFeatures))
    return newFeatures

def build_quadratic_poly(x):
    """polynomial basis functions with degree=2 but with all possible
    recombinations of features """
    numOriginalFeatures=len(x[0,:])
    numNewFeatures=np.int_((numOriginalFeatures*(numOriginalFeatures+1))/2)
#    print('numOrigFeatures:')
#    print(numOriginalFeatures)
#    print('numNewFeatures:')
#    print(numNewFeatures)
    newFeatures=np.zeros([len(x[:,0]),numNewFeatures])
    indx=0
    for i in range(numOriginalFeatures):
        for j in range(0,i+1):
            newFeatures[:,indx]=x[:,i]*x[:,j]
            indx=indx+1   
    return newFeatures

def augument_feature(x):
    """add column of ones at the beginning """
    col1=np.ones([len(x[:,0]),1])
    x=np.column_stack((col1,x))
    return x
    
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

def cross_validation(seed, numFolds, typeOfModel, y, x ):
    # build folds 
    numTrials=len(yb)
    k_indices = build_k_indices(numTrials, numFolds, seed)
    
    #test set
    x_test=x[k_indices[k,:]]
    y_test=y[k_indices[k,:]]
    
    #train set
    ind_train=[]
    for i in k_indices:
        if not i in k_indices[k,:]:
            ind_train=np.append(ind_train,i)
    ind_train=ind_train.astype(int)
    x_train=x[ind_train]
    y_train=y[ind_train]
    
    #build polynomial 
    x_test_poly=build_poly(x_test,createNewDegreeFeatures,createNewQuadraticFeatures, degree)
    x_train_poly=build_poly(x_train,createNewDegreeFeatures,createNewQuadraticFeatures, degree)
    
    #normalizing data
    x_test_norm, mean_x_test, std_x_test= standardize(x_test_poly)
    x_train_norm, mean_x_train, std_x_train= standardize(x_train_poly)
    
    # addind 1 for firs column
    x_test_norm=augument_feature(x_test_norm)
    x_train_norm=augument_feature(x_train_norm)
    
    #evaluate model using k fold crossvalidation
    RMSE_tr_all=[]
    RMSE_te_all=[]
    percDiff_tr_all=[]
    percDiff_te_all=[]
    for k in range(numFolds):
        if (leastSquaresOn==1):
            w, loss_tr=least_squares(y_train, x_train_norm)            
        if (gradientDescentOn==1):
            max_iters =500
            gamma =0.05 # 0.7
            w, loss_tr=gradient_descent(y_train,x_train_norm,max_iters,gamma,plottingOn)
        #saving loss  
        loss_te=loss_rmse(y_test, x_test_norm, w)
        RMSE_tr_all=np.append(RMSE_tr_all,loss_tr)
        RMSE_te_all=np.append(RMSE_te_all,loss_te) 
        #differences in labels
        y_pred_tr=x_train_norm.dot(w)
        y_pred_tr_1=predictionsToClasses(yb,y_pred_tr,plottingOn)
        y_pred_te=x_test_norm.dot(w)
        y_pred_te_1=predictionsToClasses(yb,y_pred_te,plottingOn)
        #calculate statistics 
        percDiff_tr=calcResultsStatistics(yb,y_pred_tr_1)  
        percDiff_te=calcResultsStatistics(yb,y_pred_te_1)    
        percDiff_tr_all=np.append(percDiff_tr_all,percDiff_tr)
        percDiff_te_all=np.append(percDiff_te_all,percDiff_te)
        
    RMSE_tr=np.mean(RMSE_tr_all)
    RMSE_te=np.mean(RMSE_te_all)
    RMSE_tr_var=np.std(RMSE_tr_all)
    RMSE_te_var=np.std(RMSE_te_all)
    percDiff_tr=np.mean(percDiff_tr_all)
    percDiff_te=np.mean(percDiff_te_all)
    percDiff_tr_var=np.std(percDiff_tr_all)
    percDiff_te_var=np.std(percDiff_te_all)
    
        

# -999 VALUES 
def resolve999values(x,createNew999Features,plottingOn):
    num_features= len(x[0,:])
    num_samples = len(x[:,0])
    
    # counting number of them 
    indx999all=[]
    for i in range(num_features):
        true999= (x[:,i]==(-999))
        indx999=np.where(true999)
        indx999all=np.append(indx999all,indx999)   
    indx999set=set(indx999all)
    print('Number of samples when at leas one feature has value -999:', len(indx999set))
    print('Percentage of total data: ', len(indx999set)/num_samples)
    
    
    # swaping -999 with mean value of that feature 
    # and creating a new features if chosen 
    newFeatures=np.zeros([num_samples,1])
    for i in range(num_features):
        true999=(x[:,i]==(-999))
        false999=(x[:,i]!=(-999))
        indxTrue999=np.where(true999)
        indxFalse999=np.where(false999)
        meanVal=np.mean(x[indxFalse999,i])
        x[indxTrue999,i]=meanVal
        #creating new features 
        newFeature=1*true999
        if(np.sum(newFeature)!=0):
            newFeatures=np.column_stack((newFeatures,newFeature))
    if (createNew999Features==1):
        newFeatures=newFeatures[:,1:]
        x=np.column_stack((x,newFeatures))
        print('Features dimension after 999 expansion:', np.shape(x))    
        
    # plotting expanded features 
    num_features2=len(x[0,:])
    if (plottingOn==1):
        fig=plt.figure(figsize=(50, 25), dpi= 40)
        for i in range(num_features2):
            ax1 = fig.add_subplot(12, 5, i+1)
            ax1.scatter(samples, xExp[:,i],marker='.');
            ax1.set(xlabel='sample', ylabel='feature value',title='Feature '+ str(i+1))
            ax1.grid()
        fig.savefig("expanded_features")
    return x

# REMOVING OUTLIERS
def removeOutliers(x, percentileOutliers):
    num_features= len(x[0,:])
    x_perc_border=np.percentile(x, percentileOutliers, axis=0)
    for i in range (num_features):
        x[:,i]=np.clip(x[:,i],0,x_perc_border[i])
    return x


def build_poly(x,createNewDegreeFeatures,createNewQuadraticFeatures, degree):
    print('---')
    print('Features dimension before polynomial basis:', np.shape(x))
    if (createNewDegreeFeatures==1):
        newFeatures1=build_poly_degree(x, degree)
    if (createNewQuadraticFeatures==1):
        newFeatures2=build_quadratic_poly(x)
    if (createNewDegreeFeatures==1):
        x=np.column_stack((x,newFeatures1))
    if (createNewQuadraticFeatures==1):
        x=np.column_stack((x,newFeatures2))
    print('Features dimension after polynomial basis:', np.shape(x))
    print('---')
    return x

def predictionsToClasses(yb,y_predicted,plottingOn):
    indx1=np.where(yb==1)
    indx_1=np.where(yb==-1)
    
    # predicted label to -1 and 1
    y_predicted_1=np.copy(y_predicted)
    y_predicted_1[np.where(y_predicted>0)]=1
    y_predicted_1[np.where(y_predicted<0)]=-1
    
    # visualizinG true and predicted labels
    if (plottingOn==1):
        fig, ax1 = plt.subplots(figsize=(50,25), dpi= 40)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(indx1,y_predicted_1[indx1],c='b')
        ax1.scatter(indx_1,y_predicted_1[indx_1],c='r')
        ax1.set(xlabel='y label', ylabel='predicted label',title='True vs predicted labels')
        ax1.grid()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(indx1,y_predicted_1[indx1],c='b')
        ax2.set(xlabel='y label', ylabel='predicted label',title=' True label = 1')
        ax2.grid()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(indx_1,y_predicted_1[indx_1],c='r')
        ax3.set(xlabel='y label', ylabel='predicted label',title='True label = -1')
        ax3.grid()
        fig.savefig("true_vs_predicted_labels_1")
    return y_predicted_1

def calcResultsStatistics(yb,y_predicted_1):
    # count number of different ones 
    indxDiff=np.where(yb!=y_predicted_1)
    indxDiff2=np.asarray(indxDiff)[0,:]
    print('Number different predictions:', len(indxDiff2))
    print('Percentage different ones:', len(indxDiff2)/len(yb))
    percDifferences=len(indxDiff2)/len(yb)
    

    #confusion matrix and error metrics
    conf_matrix, accuracy, precision, sensitivity, specificity = conf_mat(y_predicted_1, yb)
    print('Confusion matrix:', conf_matrix)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    
    return percDifferences