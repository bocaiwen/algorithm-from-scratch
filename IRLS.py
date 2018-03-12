#! /usr/bin/env python

# import tensorflow as tf
import numpy as np
from tabulate import tabulate
from scipy.special import expit

# use W to compute next R

def sigmoid(W, X):
    return expit(np.dot(X, W))


def regularize(W):
    while np.all(np.absolute(W) > 10.):
        W /= 10.
    # W won't converge to a sustained point, but the ratio of `W[0] / W[1]` do converge
    W /= np.max(np.absolute(W))
    return W * 10


def IRLS(X, Y, W, maxloop=50):
    for i in range(maxloop):
        try:
            lastW = W
            y = sigmoid(W, X)
            # print("y = {}".format(y.reshape(-1)))
            rnn = (y * (1. - y)).reshape(-1)
            # print("rnn = {}".format(rnn.reshape(-1)))
            R = np.diag(rnn)

            # use R to compute next W
            theta = X

            H_1 = np.linalg.inv(np.dot(np.dot(theta.T, R), theta))
            D = np.dot(theta.T, (sigmoid(W, X) - Y))
            t = np.dot(H_1, D)
            W = W - t
            W = regularize(W)
            print("W = {}".format(W.reshape(-1)))
            if np.all(np.round(W, 8) == np.round(lastW, 8)):
                break
            lastW = W
        except Exception as e:
            print("Error: {}".format(e))
            break

    return W
