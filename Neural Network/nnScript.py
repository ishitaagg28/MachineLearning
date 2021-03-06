import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import matplotlib.pyplot as plt



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  (1.0/(1.0+np.exp(-z)))# your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    
    train_preprocess_labelled = []
    validation_preprocess_labelled = []
    test_preprocess_labelled = []
    

    i = 1 
    p = 1
    for key in mat.keys():
        if 'train' in key:
            digit = int(key[-1])
            
            key_data = mat.get(key)
            
            label = digit*np.ones(shape=(key_data.shape[0],1))
            key_data_labelled = np.concatenate((key_data,label),axis = 1)
            
            if(i == 1):
                train_preprocess_labelled = key_data_labelled
                i+=1
            else:
                train_preprocess_labelled = np.concatenate((train_preprocess_labelled,key_data_labelled),axis = 0)
                
        else:
            if 'test' in key:
                digit = int(key[-1])
                
                t_data = mat.get(key)
                
                label = digit*np.ones(shape=(t_data.shape[0],1))
                t_data_labelled = np.concatenate((t_data,label),axis = 1)
                
                if(p == 1):
                    test_preprocess_labelled = t_data_labelled
                    p+=1
                else:
                    test_preprocess_labelled = np.concatenate((test_preprocess_labelled,t_data_labelled),axis = 0)
            
    np.random.shuffle(train_preprocess_labelled)
    
    train_data = train_preprocess_labelled[0:50000,0:784]
    train_label = train_preprocess_labelled[0:50000:,784]

    train_data = np.double(train_data) / 255.0

    validation_data = train_preprocess_labelled[50000:60000,0:784]
    validation_label   = train_preprocess_labelled[50000:60000,784]
    
    validation_data = np.double(validation_data) / 255.0
    
    
    test_data    = test_preprocess_labelled[:,0:784]
    test_label   = test_preprocess_labelled[:,784]

    test_data = np.double(test_data) / 255.0

    # Feature selection
    # Your code here.
    
    train_data_original = np.transpose(train_preprocess_labelled[:,0:784])
    exclude_cols = []
    global selected_features
    selected_features = []
    for i in range(train_data_original.shape[0]):
        if np.all(train_data_original[i] == train_data_original[i][0]):
          exclude_cols.append(i)
        else:
          selected_features.append(i)
    
    exclude_cols_bool = np.all(train_preprocess_labelled[:,0:784] == (train_preprocess_labelled[:,0:784])[0,:], axis = 0) 
    train_data      = train_data[:,~exclude_cols_bool]
    validation_data = validation_data[:,~exclude_cols_bool]
    test_data = test_data[:,~exclude_cols_bool]
    
    print('Train shape: ',train_data.shape[0],train_data.shape[1])
    print('Validation shape: ',validation_data.shape[0],validation_data.shape[1])
    print('Test shape: ',test_data.shape[0],test_data.shape[1])
    print('Selected Feature Indices: ',selected_features)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # 1: Feed Forward Propagation
    # Input Layer to Hidden Layer
    n = training_data.shape[0]
    bias = np.ones(shape=(n,1))
    input_with_bias = np.concatenate((bias, training_data),axis=1) #Adding bias term to input
    
    aj = np.dot(input_with_bias,np.transpose(w1))
    zj = sigmoid(aj)
    
    # Hidden Layer to Output Layer
    m = zj.shape[0]
    bias = np.ones(shape=(m,1))
    hidden_output_with_bias = np.concatenate((bias, zj), axis=1)
    
    bl = np.dot(hidden_output_with_bias, np.transpose(w2))
    ol = sigmoid(bl)
    
    # 2: Error function and Backpropagation
    yl = np.zeros(shape=(n,n_class)) #Creating a true output vector for each trainng sample using 1 ok K encoding
    for i in range(n):
        digit = int(training_label[i])
        yl[i][digit] = 1
        
    neg_ll_error = -1*(np.sum( np.multiply(yl,np.log(ol)) + np.multiply((1.0-yl),np.log((1.0-ol)))))/(n)
    
    grad_w2 = np.dot((ol- yl).T, hidden_output_with_bias)
   
    grad_w1 = np.dot( (np.transpose(np.dot((ol- yl),w2) * ( hidden_output_with_bias * (1.0-hidden_output_with_bias))) ), input_with_bias)
    grad_w1 = grad_w1[1:, :] # We do not require gradient of the weights at the bias hidden node
    
    # 3: Regularization in Neural Network
    obj_val = neg_ll_error + ( (lambdaval/(2*n)) * ((np.sum(w1**2) + np.sum(w2**2))) )
    grad_w1_reg = (grad_w1 + lambdaval * w1)/n
    grad_w2_reg = (grad_w2 + lambdaval * w2)/n
    obj_grad = np.concatenate((grad_w1_reg.flatten(), grad_w2_reg.flatten()),0)



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    # labels = np.array([])
    # Your code here
    
    # Input Layer to Hidden Layer
    n = data.shape[0]
    bias = np.ones(shape=(n,1))
    input_with_bias = np.concatenate((bias, data),axis=1) #Adding bias term to input
    
    aj = np.dot(input_with_bias,np.transpose(w1))
    zj = sigmoid(aj)
    
    # Hidden Layer to Output Layer
    m = zj.shape[0]
    bias = np.ones(shape=(m,1))
    hidden_output_with_bias = np.concatenate((bias, zj), axis=1)
    
    bl = np.dot(hidden_output_with_bias, np.transpose(w2))
    ol = sigmoid(bl)
    
    labels = np.argmax(ol, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20 # Optimal value obtained after performing gradient descent for a set of lambda and hidden nodes values

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

pickle.dump([selected_features, lambdaval, w1, w2, n_hidden, open('params.pickle', 'wb'))
