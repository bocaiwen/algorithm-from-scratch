#!/usr/bin/env python


import numpy as np
# import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from IRLS import IRLS, sigmoid

# data
centers0 = [[-1.,-1.], [3.,3.]]
centers1 = [[10.,10.], [15.,15.]]

cov=[[1.8, 0], [0, 1.8]]

rs = np.random.RandomState(0)

samples0 = []
for c in centers0:
    samples0.append(rs.multivariate_normal(c, cov, 30))

samples1 = []
for c in centers1:
    samples1.append(rs.multivariate_normal(c, cov, 50))

samples0 = np.concatenate(samples0)
samples1 = np.concatenate(samples1)

X = np.concatenate([samples0, samples1])

# print(samples0)
# print(samples1)

# plot
# grid = sns.JointGrid(X[:, 0], X[:, 1], space=0, size=6, ratio=50)
# grid.plot_joint(plt.scatter, color="g")
# grid.plot_marginals(sns.rugplot, height=1, color="g")
# plt.show()


# EM estimate

H = 2
N = X.shape[0]

W = np.zeros(H+1)
# pi is a logistic function on W and X

mean = np.array([[5., 5.], [7., 7.]])
# mean = np.array([[5., 5.], [7., 7.], [4., 4.], [8.,8.], [9.,9.]])
# mean = np.array([[6., 5.], [7., 9.], [4., 3.], [9.,8.]])

# add extra column 1 for linear function bias
Xx = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)

def pi(x):
    xx = np.insert(x, 0, 1.)
    p = sigmoid(W, xx)
    return np.array([p, 1. - p])


def calcR(pi, mean, X):
    def _x(x):
        pdf = np.apply_along_axis(lambda mu: multivariate_normal.pdf(x, mu, cov=cov), axis=1, arr=mean)
        pdf *= pi(x)
        total = np.sum(pdf)
        ans = pdf / total
        return ans
    #
    return np.apply_along_axis(_x, 1, X)


def calcMean(X, R, Nk):
    def _k(r):
        total = np.sum((X.T * r).T, axis=0)
        # print("1 total = {}".format(total))
        return total

    total = np.apply_along_axis(_k, axis=0, arr=R).T
    # print("2 total = {}".format(total))
    return (total.T / Nk).T


def logp(X, pi, mean):
    def _x(x):
        pdf = np.apply_along_axis(lambda mu: multivariate_normal.pdf(x, mu, cov=cov), axis=1, arr=mean)
        pdf *= pi(x)
        total = np.sum(pdf)
        return np.log(total)
    #
    return np.sum(np.apply_along_axis(_x, 1, X))


# calc

loop = 10

for i in range(loop):
    # E-step

    R = calcR(pi, mean, X)
    # print(R)

    # M-step

    Nk = np.sum(R, axis=0)
    print("Nk =\n{}".format(Nk))

    # pi = Nk/N
    # As we now support only binomial logistic regression, passing the first column of R is OK
    W = IRLS(Xx, R[:, 0], W)
    print("W =\n{}".format(W))


    mean = calcMean(X, R, Nk)
    print("mean =\n{}".format(mean))

    logp_ = logp(X, pi, mean)
    print("logp_ =\n{}".format(logp_))


pix = np.apply_along_axis(pi, 1, X)
print("pix =\n{}".format(pix))
