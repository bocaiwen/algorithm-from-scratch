#! /usr/bin/env python

# import tensorflow as tf
import numpy as np
from tabulate import tabulate
from scipy.special import expit
from IRLS import IRLS, sigmoid

def X2Train(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X.reshape(-1, 1)], axis=1)


if __name__ == '__main__':

    X = np.array([1,2,1,2,5,9,8,9,8,9], dtype=np.float32)
    # X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
    Y = np.array([1,1,1,1,1,0,0,0,0,0], dtype=np.float32)

    X = X2Train(X)
    Y = Y.reshape(-1, 1)


    # W = np.random.normal((2,1)).reshape(-1, 1)
    # W = np.array([-4.37665939, 0.89719141]).reshape(-1, 1)
    W = np.array([0.,0.]).reshape(-1, 1)

    W = IRLS(X, Y, W)
    y = sigmoid(W, X)
    print("y =\n{}".format(y))
    print("real Y = \n{}".format(Y))
