import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones(shape=(n_data,1))
    input_with_bias = np.concatenate((bias, train_data),axis=1) #Adding bias term to input

    initialWeights = initialWeights.reshape(n_features+1,1)

    theta = sigmoid(np.dot(input_with_bias,initialWeights))

    error = -1.0*(np.sum( np.multiply(labeli,np.log(theta)) + np.multiply((1.0-labeli),np.log((1.0-theta)))))/(n_data)

    error_grad = (np.sum( np.multiply((theta-labeli),input_with_bias), axis = 0))/(n_data)

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones(shape=(data.shape[0],1))
    input_with_bias = np.concatenate((bias, data),axis=1) #Adding bias term to input

    ##y_pred = sigmoid(np.dot(input_with_bias,W))

    y_pred = np.argmax(sigmoid(np.dot(input_with_bias,W)),axis=1)
    label = y_pred.reshape(data.shape[0],1)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones(shape=(n_data,1))
    input_with_bias = np.concatenate((bias, train_data),axis=1) #Adding bias term to input

    weights = params.reshape(n_feature+1,n_class)

    theta_numerator = np.exp(np.dot(input_with_bias,weights))
    theta_denominator = np.sum(np.exp(np.dot(input_with_bias,weights)),axis = 1, keepdims = True)

    theta =  theta_numerator/theta_denominator

    error = (-1)*(np.sum(np.sum(np.multiply(labeli,np.log(theta)))))
    error_grad = (np.dot(np.transpose(input_with_bias),(theta - labeli))).ravel()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones(shape=(data.shape[0],1))
    input_with_bias = np.concatenate((bias, data),axis=1) #Adding bias term to input
    
    theta_value = np.exp(np.dot(input_with_bias,W))/(np.sum(np.exp(np.dot(input_with_bias,W)),axis=1, keepdims = True))
    
    label = (np.argmax(theta_value,axis=1)).reshape(data.shape[0],1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label_train = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_validation = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_validation == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_test = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')

## Calculating Error and Accuracies for One-vs-Rest Logistic Regression with respect to every category (0-9) 
## of the train and test data
print("Classwise Statistics for One-vs-Rest Logistic Regression on Training Data")
temp = 0
input_mat = loadmat('mnist_all.mat')
n_validation = 1000
for i in range(10):
    size_i = input_mat.get("train" + str(i)).shape[0]

    print("\n Statistics for class: ",i)
    print("\t Count of samples: ", size_i - n_validation)
    print("\t Count of matches: ", np.sum((predicted_label_train[temp:temp + size_i - n_validation,:] == train_label[temp:temp + size_i - n_validation,:]).astype(float)))
    print("\t Count of mismatches: ", np.sum((predicted_label_train[temp:temp + size_i - n_validation,:] != train_label[temp:temp + size_i - n_validation,:]).astype(float)))
    print('\t Training set Accuracy:' + str(100 * np.mean((predicted_label_train[temp:temp + size_i - n_validation,:] == train_label[temp:temp + size_i - n_validation,:]).astype(float))) + '%')
    print('\t Training set Error:' + str(100 * np.mean((predicted_label_train[temp:temp + size_i - n_validation,:] != train_label[temp:temp + size_i - n_validation,:]).astype(float))) + '%')


    temp = temp + size_i - n_validation

print("Classwise Statistics for One-vs-Rest Logistic Regression on Test Data")
temp = 0
for i in range(10):
    size_i = input_mat.get("test" + str(i)).shape[0]

    print("\n Test Statistics for class: ",i)
    print("\t Count of samples: ", size_i)
    print("\t Count of matches: ", np.sum((predicted_label_test[temp:temp + size_i,:] == test_label[temp:temp + size_i,:]).astype(float)))
    print("\t Count of mismatches: ", np.sum((predicted_label_test[temp:temp + size_i,:] != test_label[temp:temp + size_i,:]).astype(float)))
    print('\t Test set Accuracy:' + str(100 * np.mean((predicted_label_test[temp:temp + size_i,:] == test_label[temp:temp + size_i,:]).astype(float))) + '%')
    print('\t Test set Error:' + str(100 * np.mean((predicted_label_test[temp:temp + size_i,:] != test_label[temp:temp + size_i,:]).astype(float))) + '%')


    temp = temp + size_i

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #

## Randomly selecting 10000 training samples to train the SVM
np.random.seed(9)
train_indices = np.random.choice(train_data.shape[0], size=10000, replace=False)

svm_train_data = train_data[train_indices,:]
svm_train_label = train_label[train_indices,:]

## Learning SVM Using linear kernel (all other parameters are kept default).
linear_SVM = svm.SVC(kernel = "linear")
linear_SVM.fit(svm_train_data,svm_train_label.flatten())

print("Accuracy and Error for Linear kernel SVM:")
print('\n Training set Accuracy:' + str(100 * linear_SVM.score(svm_train_data,svm_train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * linear_SVM.score(validation_data,validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * linear_SVM.score(test_data,test_label)) + '%')

## Using radial basis function with value of gamma setting to 1 (all other parameters are kept default).
rbf_SVM_gamma1 = svm.SVC(kernel = "rbf",gamma = 1.0)
rbf_SVM_gamma1.fit(svm_train_data,svm_train_label.flatten())

print("Accuracy and Error for radial basis function with value of gamma setting to 1:")
print('\n Training set Accuracy:' + str(100 * rbf_SVM_gamma1.score(svm_train_data,svm_train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * rbf_SVM_gamma1.score(validation_data,validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * rbf_SVM_gamma1.score(test_data,test_label)) + '%')

## Using radial basis function with value of gamma setting to default (all other parameters are kept
## default).
rbf_SVM_gamma_default = svm.SVC(kernel = "rbf")
rbf_SVM_gamma_default.fit(svm_train_data,svm_train_label.flatten())

print("Accuracy and Error for radial basis function with value of gamma setting to default:")
print('\n Training set Accuracy:' + str(100 * rbf_SVM_gamma_default.score(svm_train_data,svm_train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * rbf_SVM_gamma_default.score(validation_data,validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * rbf_SVM_gamma_default.score(test_data,test_label)) + '%')

## Using radial basis function with value of gamma setting to default and varying value of C (1, 10, 20, 30,...., 100)

c_values = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
train_accuracy_store = np.zeros(len(c_values),float)
test_accuracy_store = np.zeros(len(c_values),float)
validation_accuracy_store = np.zeros(len(c_values),float)
index = 0

print("Accuracy and Error for SVM with radial basis function and value of gamma setting to default and varying C values:")

for c in c_values:
  rbf_SVM_c = svm.SVC(kernel = "rbf" , C = c)
  rbf_SVM_c.fit(svm_train_data,svm_train_label.flatten())
  train_accuracy_store[index] = 100 * rbf_SVM_c.score(svm_train_data,svm_train_label)
  validation_accuracy_store[index] = 100 * rbf_SVM_c.score(validation_data,validation_label)
  test_accuracy_store[index] = 100 * rbf_SVM_c.score(test_data,test_label)

  print('Accuracy for C = ' + str(c))
  print('\n Training set Accuracy:' + str(train_accuracy_store[index]) + '%')
  print('\n Validation set Accuracy:' + str(validation_accuracy_store[index]) + '%')
  print('\n Testing set Accuracy:' + str(test_accuracy_store[index]) + '%')

  index = index + 1

## Plot of Accuracy vs C
fig = plt.figure(figsize=[12,6])
plt.plot(c_values,train_accuracy_store,color = 'r')
plt.plot(c_values,validation_accuracy_store,color = 'g')
plt.plot(c_values,test_accuracy_store,color = 'b')
plt.title('Accuracy vs C for SVM with Radial Basis Function')
plt.legend(['Train data','Validation Data','Test Data'])

plt.xlabel('C', weight='bold')
plt.ylabel('Accuracy', weight='bold')

plt.xticks( c_values)

plt.show()

## Using radial basis function with value of gamma setting to auto and varying value of C (1, 10, 20, 30,...., 100)

c_values = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
train_accuracy_store = np.zeros(len(c_values),float)
test_accuracy_store = np.zeros(len(c_values),float)
validation_accuracy_store = np.zeros(len(c_values),float)
index = 0

print("Accuracy and Error for SVM with radial basis function and value of gamma setting to 'auto' and varying C values:")

for c in c_values:
  rbf_SVM_c = svm.SVC(kernel = "rbf" , C = c, gamma = "auto")
  rbf_SVM_c.fit(svm_train_data,svm_train_label.flatten())
  train_accuracy_store[index] = 100 * rbf_SVM_c.score(svm_train_data,svm_train_label)
  validation_accuracy_store[index] = 100 * rbf_SVM_c.score(validation_data,validation_label)
  test_accuracy_store[index] = 100 * rbf_SVM_c.score(test_data,test_label)

  print('Accuracy for C = ' + str(c))
  print('\n Training set Accuracy:' + str(train_accuracy_store[index]) + '%')
  print('\n Validation set Accuracy:' + str(validation_accuracy_store[index]) + '%')
  print('\n Testing set Accuracy:' + str(test_accuracy_store[index]) + '%')

  index = index + 1

## Plot of Accuracy vs C
fig = plt.figure(figsize=[12,6])
plt.plot(c_values,train_accuracy_store,color = 'r')
plt.plot(c_values,validation_accuracy_store,color = 'g')
plt.plot(c_values,test_accuracy_store,color = 'b')
plt.title('Accuracy vs C for SVM with Radial Basis Function')
plt.legend(['Train data','Validation Data','Test Data'])

plt.xlabel('C', weight='bold')
plt.ylabel('Accuracy', weight='bold')

plt.xticks( c_values)

plt.show()

## Learning SVM Using rbf kernel, c = 10 (all other parameters are kept default).
rbf_SVM_best_params = svm.SVC(kernel = "rbf", C = 10.0)
rbf_SVM_best_params.fit(train_data,train_label.flatten())

print("Accuracy and Error for SVM Using rbf kernel, c = 10 and all other parameters are kept default:")
print('\n Training set Accuracy:' + str(100 * rbf_SVM_best_params.score(train_data,train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * rbf_SVM_best_params.score(validation_data,validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * rbf_SVM_best_params.score(test_data,test_label)) + '%')

## Learning SVM Using rbf kernel, c = 10 (all other parameters are kept default).
rbf_SVM_best_params = svm.SVC(kernel = "rbf", C = 10.0, gamma="auto")
rbf_SVM_best_params.fit(train_data,train_label.flatten())

print("Accuracy and Error for SVM Using rbf kernel, c = 10 and all other parameters are kept default:")
print('\n Training set Accuracy:' + str(100 * rbf_SVM_best_params.score(train_data,train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * rbf_SVM_best_params.score(validation_data,validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * rbf_SVM_best_params.score(test_data,test_label)) + '%')


##################


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

print("\n Accuracy and Error for Multi-class Logistic Regression:")

# Find the accuracy on Training Dataset
predicted_label_b_train = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b_val = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b_val == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b_test = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b_test == test_label).astype(float))) + '%')


## Calculating Error and Accuracies for Multi-class Logistic Regression with respect to every category (0-9) 
## of the train and test data
print("Classwise Statistics for Multi-Class Logistic Regression on Training Data")
temp = 0
input_mat = loadmat('mnist_all.mat')
n_validation = 1000
for i in range(10):
    size_i = input_mat.get("train" + str(i)).shape[0]

    print("\n Statistics for class: ",i)
    print("\t Count of samples: ", size_i - n_validation)
    print("\t Count of matches: ", np.sum((predicted_label_b_train[temp:temp + size_i - n_validation,:] == train_label[temp:temp + size_i - n_validation,:]).astype(float)))
    print("\t Count of mismatches: ", np.sum((predicted_label_b_train[temp:temp + size_i - n_validation,:] != train_label[temp:temp + size_i - n_validation,:]).astype(float)))
    print('\t Training set Accuracy:' + str(100 * np.mean((predicted_label_b_train[temp:temp + size_i - n_validation,:] == train_label[temp:temp + size_i - n_validation,:]).astype(float))) + '%')
    print('\t Training set Error:' + str(100 * np.mean((predicted_label_b_train[temp:temp + size_i - n_validation,:] != train_label[temp:temp + size_i - n_validation,:]).astype(float))) + '%')


    temp = temp + size_i - n_validation

print("Classwise Statistics for Multi-Class Logistic Regression on Test Data")
temp = 0
for i in range(10):
    size_i = input_mat.get("test" + str(i)).shape[0]

    print("\n Test Statistics for class: ",i)
    print("\t Count of samples: ", size_i)
    print("\t Count of matches: ", np.sum((predicted_label_b_test[temp:temp + size_i,:] == test_label[temp:temp + size_i,:]).astype(float)))
    print("\t Count of mismatches: ", np.sum((predicted_label_b_test[temp:temp + size_i,:] != test_label[temp:temp + size_i,:]).astype(float)))
    print('\t Test set Accuracy:' + str(100 * np.mean((predicted_label_b_test[temp:temp + size_i,:] == test_label[temp:temp + size_i,:]).astype(float))) + '%')
    print('\t Test set Error:' + str(100 * np.mean((predicted_label_b_test[temp:temp + size_i,:] != test_label[temp:temp + size_i,:]).astype(float))) + '%')


    temp = temp + size_i
