import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    Y_Labels = np.unique(y).tolist()
    means = np.full((X.shape[1],len(Y_Labels)), float(0))
    
    for i in range(0,len(Y_Labels)):
        label = Y_Labels[i]
        row_count = 0
        sum_of_arrays = np.zeros(X.shape[1])
        for j in range(0,len(X)):
            if(y[j]==label):
                row_count = row_count+1
                
                sum_of_arrays = sum_of_arrays + X[j]
                    
                for d in range(0,X.shape[1]):
                    means[d][i] = sum_of_arrays[d]/row_count
                    
        
                     
    covmat = np.cov(X.transpose())
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    Y_Labels = np.unique(y).tolist()
    means = np.full((X.shape[1],len(Y_Labels)), float(0))
    
    covmats = []
    
    for i in range(0,len(Y_Labels)):
        label = Y_Labels[i]
        row_count = 0
        covmat_label = []
        sum_of_arrays = np.zeros(X.shape[1])
        for j in range(0,len(X)):
            if(y[j]==label):
                row_count = row_count+1
                covmat_label.append(X[j].tolist())
                
                sum_of_arrays = sum_of_arrays + X[j]
                    
                for d in range(0,X.shape[1]):
                    means[d][i] = sum_of_arrays[d]/row_count
                    
        covmats.append(np.array(covmat_label))
        covmats[i] = np.cov(covmats[i].transpose())
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means = means.transpose()
    count_match = 0
    ypred = np.full((len(Xtest),1), 0)
    
    for i in range(0,len(Xtest)):
        ypred_temp = np.zeros(len(means))
        
        for j in range(0,len(means)):
            m_distance = np.matmul(np.matmul((Xtest[i]-means[j]).transpose(),np.linalg.inv(covmat)),(Xtest[i]-means[j]))
            
            ypred_temp[j] = (np.exp((-0.5)*(m_distance)))/(np.linalg.det(covmat))
        
        
        ypred[i] = np.argmax(ypred_temp)+1
        if(ytest[i]==ypred[i]):
            count_match = count_match + 1
    
    acc = (count_match/len(ytest))*100
    
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    means = means.transpose()
    count_match = 0
    ypred = np.full((len(Xtest),1), 0)
    
    for i in range(0,len(Xtest)):
        ypred_temp = np.zeros(len(means))
        
        for j in range(0,len(means)):
            m_distance = np.matmul(np.matmul((Xtest[i]-means[j]).transpose(),np.linalg.inv(covmats[j])),(Xtest[i]-means[j]))
            
            ypred_temp[j] = (np.exp((-0.5)*(m_distance)))/(np.linalg.det(covmats[j]))
        
        
        ypred[i] = np.argmax(ypred_temp)+1
        if(ytest[i]==ypred[i]):
            count_match = count_match + 1
    
    acc = (count_match/len(ytest))*100
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    sq_X = np.dot(X.transpose(),X)
    inv_sq_X = np.linalg.inv(sq_X)
    trans_X_dot_y = np.dot(X.transpose(),y)
    w = np.dot(inv_sq_X,trans_X_dot_y)
    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    new_X = np.dot(X.transpose(),X) 
    identity_matrix_lambda = lambd * np.identity(np.shape(X)[1]) 
    inv_mat = np.linalg.inv(new_X + identity_matrix_lambda)
    trans_X_dot_y = np.dot(X.transpose(),y)
    w = np.dot(inv_mat,trans_X_dot_y)
    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    N = np.shape(Xtest)[0]
    squared_loss = np.power(ytest - np.dot(Xtest,w),2)
    sum_of_squared_loss = np.sum(squared_loss)
    mse = sum_of_squared_loss/N
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    
    w = w.reshape(65,1)
    
    loss = y - np.dot(X,w)
    loss_trans = loss.transpose()
    error = 0.5 * (np.dot(loss_trans,loss) + lambd * np.dot(w.transpose(),w))
    error = error.flatten()
    
    error_grad = np.dot(-1*(X.transpose()),loss)+ lambd *w
    error_grad = error_grad.flatten()
    error = error.flatten()
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1))
    # IMPLEMENT THIS METHOD
    
    Xp = np.ones((x.shape[0],p+1))    
    
    for i in range(0,len(x)):
        for j in range(1,p+1):
            Xp[i][j] = x[i]**j
                     
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
#lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
lambda_opt = lambdas[np.argmin(mses3)]
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
