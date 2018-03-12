#! /usr/bin/env python

# BRML Example 20.2

# import tensorflow as tf
import numpy as np
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp


MixBernoulli = pd.read_csv('MixBernoulli.csv', header=None)
data = MixBernoulli.values
h_true = pd.read_csv('MixBernoulli-trueH.csv', header=None).values
# print("h_true.shape = {}".format(h_true.shape))
h_true = h_true - 1
h_true = h_true.reshape(-1)

H = 2
loop = 1000
N, D = data.shape

def mixBernoulli(data, H, loop):
    N, D = data.shape

    ph = np.random.rand(H)
    ph /= np.sum(ph)

    qh = np.zeros((N, H))
    pvh = np.random.rand(D, H)
    for d in range(D):
        pvh[d, :] /= np.sum(pvh[d, :])

    ph1s = []
    for i in range(loop):
        # E-step
        tqh = np.zeros((N, H))
        for h in range(H):
            tqh[:, h] = np.log(ph[h])
            for d in range(D):
                tqh[:, h] += np.log((data[:, d] == 1.) * pvh[d, h] + (data[:, d] == 0.) * (1 - pvh[d, h]) + (data[:, d] == 0.5) * 1.)
        tqhtotal = logsumexp(tqh, axis=1)
        for h in range(H):
            qh[:, h] = np.exp(tqh[:, h] - tqhtotal)
        # M-step
        for d in range(D):
            for h in range(H):
                pvh[d, h] = np.dot((data[:, d] == 1.), qh[:, h]) / (np.dot((data[:, d] == 1.), qh[:, h]) + np.dot((data[:, d] == 0.), qh[:, h]))
        #
        phtotal = 0.
        for h in range(H):
            ph[h] = np.sum(qh[:, h])
            phtotal += ph[h]
        for h in range(H):
            ph[h] = ph[h] / phtotal
        ph1s.append(ph[0])
    return ph1s, qh

ph1s, qh = mixBernoulli(data, H, loop)

# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
sns.pointplot(x=range(loop), y=ph1s, ax=ax1)
# plt.show()
sns.barplot(x=range(N), y=qh[:, 1], ax=ax2)
# plt.show()
sns.barplot(x=range(N), y=h_true, ax=ax3)
plt.show()
