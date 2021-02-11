# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:39:05 2020
2_Layer_Neural_Network

In this script, In this notebook, all functions required to build a 2-layer
all the functions required to build a deep neural network are implemented.

Most of these functions are based on a deep neural network script from
Andrew Ng's Deep Learning specialization in coursera. However, in the course
the neural network was supposed to solve an image classification problem.

In this implementaion, this neural network works with a non-binary output.

Data is obtained from an Excel table containing data from multiple sensors and
a single output

@author: ecastaneda
"""
import pandas as pd
import numpy as np
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import matplotlib.pyplot as plt
from general_utils import normalizar, desnormalizar, maxp, minp


def DataPreparation(file_name,pctg_test):
    '''
    This scripts reads an Excel file and generates train and test sets.
    Data processing includes following steps:
            - Read data
            - Normalize data
            - Scramble data
            - Separated data into train and test sets 
              (input and output separated)
    
    Parameters
    ----------
    file_name : string
        Name of the excel file from where the NN is getting data. 
    pctg_test : float
        Value from 0 -1 indicating the rate of distribution of test and
        training sets. The value indicated is the porcetage of total samples
        that will be inclued in test sets

    Returns
    -------
    train_x : dataframe
        input data, of shape (n_x, number of examples).
    train_y : dataframe
        output data, of shape (n_y, number of examples)..
    test_x : dataframe
        input data, of shape (n_x, number of examples).
    test_y : dataframe
        output data, of shape (n_y, number of examples).

    '''
    # Reads DB (inputs and output)
    df = pd.read_excel(file_name)
    # Get Max and Min to denormalize data 
    MRange = [df.max(), df.min()]
    # Normalize each column of the df
    normalized_df=(df-df.min())/(df.max()-df.min())
    # Data Scrambling
    scrambled_df = normalized_df.sample(frac=1)
    
    # Mask to separate data into train and test sample
    msk = np.random.rand(len(scrambled_df)) < pctg_test
    # Generate train and test data set (including inputs and outputs)
    train = scrambled_df[~msk]
    test = scrambled_df[msk]
    
    # Separates inputs and outputs train set
    train_x = train.iloc[:, 0:-1]
    train_y = train.iloc[:, -1]
    train_y = train_y.to_frame()

    # Transpose train data (shape: nx/ny, number_of_samples)
    train_x = train_x.T 
    train_y = train_y.transpose()

    # Separates inputs and outputs train set
    test_x = test.iloc[:, 0:-1]
    test_y = test.iloc[:, -1]
    test_y = test_y.to_frame()
    
    # Transpose test data (shape: nx/ny, number_of_samples)
    test_x = test_x.T 
    test_y = test_y.transpose()
    
    return train_x, train_y, test_x, test_y, MRange


def initialize_parameters(n_x, n_h, n_y):
    """
    
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous
                                                           layer, 
                                                           number of examples)
    W -- weights matrix: numpy array of shape (size of current layer,
                                               size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation 
         parameter 
    cache -- a python tuple containing "A", "W" and "b" ; 
             stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): 
              (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer,
                                               size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a
                  text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation
         value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    # if len(activation_cache.shape) == 1:
    #    activation_cache = activation_cache.reshape((-1,1))
    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL, Y):
    """
    Implement the cost function (MSE - Mean Square Error)

    Arguments:
    AL -- probability vector corresponding to your label predictions,
          shape (1, number of examples)
    Y -- output vector, shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    squared_errors=(AL-Y)**2
    cost = (1/m)*np.sum(squared_errors, axis = 1)
    
    # To make sure your cost's shape is what we expect 
    # (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single 
    layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output 
          (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation 
               (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
          same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
          same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for
             computing backward propagation efficiently
    activation -- the activation to be used in this layer, 
                  stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation 
               (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
          same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
          same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ =  sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]- learning_rate*grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]- learning_rate*grads["db" + str(l + 1)]
    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate = 0.25, 
                    num_iterations = 5000, print_cost=False):
    """
    Implements a two-layer neural network
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- output data, of shape(1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the
    # functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation
        # Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "relu")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = (2/m)*np.sum(A2-Y.to_numpy(), axis = 0)
        if len(dA2.shape) == 1:
            dA2 = dA2.reshape((1,-1))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1".
        # Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "relu")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    print("Cost after iteration {}: {}".format(num_iterations, 
                                                   np.squeeze(cost)))
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- input dataset (shape: n_x, number of examples)
    y -- output dataset (shape: n_y, number of examples)
    parameters -- parameters of the trained model
    
    Returns 
    probas -- Generated output array of forward propagation using trained model
    """
    
    #m = X.shape[1]
    
    # Forward propagation
    A1, cache1 = linear_activation_forward(X, parameters['W1'],
                                           parameters['b1'], "relu") 
    probas, cache2 = linear_activation_forward(A1, parameters['W2'],
                                               parameters['b2'], "relu")
    
    plt.plot(y.values.T,"b", probas.T,"r")
    plt.legend(["Real", "Generated"])
    plt.ylabel('Value')
    plt.xlabel('Sample')
    plt.title("Real vs Prediction")
    plt.show()
    
    return probas


# FUNCION PRINCIPAL
# test dataset 'datasetNN.xls'
train_x, train_y, test_x, test_y, MRange = DataPreparation('BDA2.xlsx',0.2)
n_x = 6
n_h = 12
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y),
                             learning_rate = 0.75,
                             num_iterations = 35000, print_cost=True)
nn_out_train  = predict(train_x, train_y, parameters)
nn_out_test  = predict(test_x, test_y, parameters)