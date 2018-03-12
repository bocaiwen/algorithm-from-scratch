import numpy as np
from anytree import NodeMixin
from lca import LCA

# Hierachical latent class analysis
class HLCA(NodeMixin):

    def __init__(self, lca, data, Y, upperClass, classThreshold, parent=None):
        super(HLCA, self).__init__()
        self.lca = lca
        self.nclass = lca.nclass
        self.data = data
        self.Y = Y
        self.upperClass = upperClass
        self.classThreshold = classThreshold
        self.parent = parent

        if self.Y is not None:
            self.ratio, self.subdata, self.subY = self.getSurviveRatio(lca, Y)
            print("ratio: {}, subdata: {}".format(self.ratio, [len(sd) for sd in self.subdata]))
        else:
            self.subdata = self.getSubdata()
            print("subdata: {}".format([len(sd) for sd in self.subdata]))

    def getSurviveRatio(self, lca, Y):
        """
        """
        ratio = np.zeros(lca.nclass)
        subdata = range(lca.nclass)
        subY = range(lca.nclass)
        for h in range(lca.nclass):
            yh = Y[ lca.qh[:, h] > self.classThreshold ]
            ratio[h] = sum(yh) / (float(len(yh)) + 1.192e-7)
            subdata[h] = self.data[ lca.qh[:, h] > self.classThreshold ]
            subY[h] = yh

        return ratio, subdata, subY

    def getSubdata(self):
        subdata = range(self.lca.nclass)
        for h in range(self.lca.nclass):
            subdata[h] = self.data[ self.lca.qh[:, h] > self.classThreshold ]

        return subdata


def expand(hlca, nclass=2, n_values=2, min_ratio=0.75, min_split=3, max_level=2):
    if max_level == 0:
        return

    ratio, subdata = hlca.ratio, hlca.subdata

    for h in range(hlca.nclass):
        if ratio[h] < min_ratio and ratio[h] > (1 - min_ratio) and len(subdata[h]) > min_split:
            lca = LCA(max_nclass=nclass, n_values=n_values)
            lca.fit(subdata[h])
            if lca.bic >= (hlca.lca.bic - 1):
                continue

            chlca = HLCA(lca, subdata[h], hlca.subY[h], h, hlca.classThreshold, hlca)
            expand(chlca, nclass, n_values, min_ratio, min_split, max_level - 1)


def expandWithoutRatio(hlca, nclass=2, n_values=2, min_split=3, max_level=2):
    if max_level == 0:
        return

    for h in range(hlca.nclass):
        if len(hlca.subdata[h]) > min_split:
            lca = LCA(max_nclass=nclass, n_values=n_values)
            lca.fit(hlca.subdata[h])
            if lca.bic >= (hlca.lca.bic - 1):
                continue
            
            chlca = HLCA(lca, hlca.subdata[h], None, h, hlca.classThreshold, hlca)
            expandWithoutRatio(chlca, nclass, n_values, min_split, max_level - 1)


def predictWithoutRatioSingle(newdata, hlca):
    _, logp = hlca.lca.predict(newdata.reshape(1, -1))

    for c in hlca.children:
        c_logp = predictWithoutRatioSingle(newdata, c)
        if c_logp > logp:
            logp = c_logp

    return logp


def predictWithoutRatio(newdata, hlca0, hlca1):
    ret = []
    for d in newdata:
        logp0 = predictWithoutRatioSingle(d, hlca0)
        logp1 = predictWithoutRatioSingle(d, hlca1)
        ret.append( 1 if logp1 > logp0 else 0)

    return np.array(ret)


# predict a single sample
def predictSingle(newdata, hlca):
    """
    """
    # print("newdata = {}".format(newdata))
    qh, _ = hlca.lca.predict(newdata.reshape(1, -1))
    # print("qh = {}".format(qh))
    # print("ratio = {}".format(hlca.ratio))
    h = np.argmax(qh)
    h2c = dict([(c.upperClass, c) for c in hlca.children])
    if h2c.get(h) is None:
        return hlca.ratio[h]
    #
    return predictSingle(newdata, h2c.get(h))


def predict(newdata, hlca):

    return np.apply_along_axis(predictSingle, 1, newdata, hlca)


def predictSingleOverTree(newdata, hlca):
    """
    Using the largest logp node to predict survive ratio.
    """
    qh, logp = hlca.lca.predict(newdata.reshape(1, -1))
    h = np.argmax(qh)
    ratio = hlca.ratio[h]

    for c in hlca.children:
        c_ratio, c_logp = predictSingleOverTree(newdata, c)
        if c_logp > logp:
            logp = c_logp
            ratio = c_ratio

    return ratio, logp


def predictOverTree(newdata, hlca):

    return np.apply_along_axis(predictSingleOverTree, 1, newdata, hlca)
