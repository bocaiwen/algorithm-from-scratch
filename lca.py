#

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logsumexp
from types import IntType


# Get posterior class probability
def posteriorOfHiddenClass(data, H, ph, pvh):
    N, D = data.shape

    tqh = np.zeros((N, H))
    tqhtotal = np.zeros(N)
    for h in range(H):
        tqh[:, h] = np.log(ph[h])
        for d in range(D):
            tqh[:, h] += np.log((data[:, d] == 1.) * pvh[d, h] + (data[:, d] == 0.) * (1 - pvh[d, h]) + (data[:, d] == 0.5) * 1. + 1.192e-7)
    tqhtotal = logsumexp(tqh, axis=1)
    # print("tqhtotal = {}".format(tqhtotal))
    qh = np.zeros((N, H))
    for h in range(H):
        qh[:, h] = np.exp(tqh[:, h] - tqhtotal)

    return qh, np.sum(tqhtotal)

#
def mixBernoulli(data, H, loop):
    N, D = data.shape

    ph = np.random.rand(H)
    ph /= np.sum(ph)

    qh = np.zeros((N, H))
    pvh = np.random.rand(D, H)
    for d in range(D):
        pvh[d, :] /= np.sum(pvh[d, :])

    phs = []
    logp = 0.
    for i in range(loop):
        # E-step
        qh, logp = posteriorOfHiddenClass(data, H, ph, pvh)
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
        # print("ph = {}".format(ph))
    return np.array(phs), pvh, ph, qh, logp


# One level of latent class analysis
class LCA(object):

    def __init__(self, max_nclass=2, max_loop=50, n_values=9):
        self.max_nclass = max_nclass
        self.nclass = max_nclass
        self.max_loop = max_loop
        self.n_values = n_values
        self.onehot = OneHotEncoder(n_values=self.n_values, sparse=False)

    def fit(self, data, trycount=10):
        self.data = data
        self.onehot.fit(self.data)
        self.hotdata = self.onehot.transform(self.data)
        # 
        bestbic = np.inf
        for nc in range(2, self.max_nclass + 1):
            print("nclass = {}".format(nc))
            for i in range(trycount):
                phs, pvh, ph, qh, logp = mixBernoulli(self.hotdata, nc, self.max_loop)
                bic = self._BIC(nc, logp)
                if bic < bestbic:
                    self.phs, self.pvh, self.ph, self.qh, self.logp = phs, pvh, ph, qh, logp
                    bestbic = bic
                    self.nclass = nc
                    self.bic = bic
                    print("bestbic = {}".format(bestbic))

    def predict(self, newdata):
        newhotdata = self.onehot.transform(newdata)

        qh, logp = posteriorOfHiddenClass(newhotdata, self.nclass, self.ph, self.pvh)
        return qh, logp

    def sampleByClass(self, h):
        pvs = np.split(self.pvh[:, h], self.onehot.feature_indices_[1:])
        sample = [np.random.multinomial(1, pv) for pv in pvs[:-1]]
        origfeat = np.array([np.argmax(s) for s in sample])
        return origfeat

    def _BIC(self, nclass, logp):
        def _freeParameter(nclass):
            return nclass + nclass * np.sum(self.n_values) * 0.3

        return np.log(len(self.data)) * _freeParameter(nclass) - 2 * logp
