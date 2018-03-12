#! /usr/bin/env python

# BRML Example 20.2

# import tensorflow as tf
import numpy as np
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp


MixBernoulli = pd.read_csv('MixBernoulliDigit.csv', header=None)
data = MixBernoulli.values

H = 20
N, D = data.shape
loop = 20

def mixBernoulli(data, H, loop):
    N, D = data.shape

    ph = np.random.rand(H)
    ph /= np.sum(ph)

    qh = np.zeros((N, H))
    pvh = np.random.rand(D, H)
    for d in range(D):
        pvh[d, :] /= np.sum(pvh[d, :])

    phs = []
    for i in range(loop):
        # E-step
        tqh = np.zeros((N, H))
        for h in range(H):
            tqh[:, h] = np.log(ph[h])
            for d in range(D):
                tqh[:, h] += np.log((data[:, d] == 1.) * pvh[d, h] + (data[:, d] == 0.) * (1 - pvh[d, h]) + (data[:, d] == 0.5) * 1. + 1.192e-7)
        tqhtotal = logsumexp(tqh, axis=1)
        print("tqhtotal = {}".format(tqhtotal))
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
        phs.append(ph)
        print("ph = {}".format(ph))
    return np.array(phs), pvh

# phs, pvh = mixBernoulli(data, H, loop)
from lca import LCA
dlca = LCA(H,loop,2)
dlca.fit(data)
phs = dlca.phs
pvh = dlca.pvh
ph = phs[-1, :]
print("final ph = {}".format(ph))

# Set up the matplotlib figure
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
# sns.pointplot(x=range(loop), y=phs[:, 0], ax=ax1)
# sns.pointplot(x=range(loop), y=phs[:, 1], ax=ax2)
# sns.pointplot(x=range(loop), y=phs[:, 2], ax=ax3)
# plt.show()

f, axs = plt.subplots(4, 5)
for h in range(H):
    # sample = np.random.binomial(1, pvh[:, h])
    sample = dlca.sampleByClass(h)
    img = sample.reshape((28, 28))
    axs[int(h/5), int(h%5)].imshow(img)

plt.show()


# posterior

newdata = data[0:1, :]
pred = dlca.predict(newdata)
pred = np.argmax(pred, axis=1)
print("Predict: ")
print(pred)


d1 = data[0, :]
plt.imshow(d1.reshape((28,28)))
plt.show()
