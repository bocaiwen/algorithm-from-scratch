#!/usr/bin/env python

from __future__ import print_function
from lca import LCA
from hierachicalLCA import HLCA, expand
from anytree import RenderTree

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    mog = pd.read_csv('mixture-of-gaussian.csv', header=None)
    mog = np.round(mog)
    mog = mog.astype(int)
    mog = mog + 5
    # print(mog)
    # print(np.max(mog, axis=0))

    lca = LCA(max_nclass=2, n_values=24)
    lca.fit(mog.values)

    print("lca.qh =\n{}".format(lca.qh))
    print("lca.logp =\n{}".format(lca.logp))

    Y = np.array([i % 2 for i in range(len(mog))])
    Y = Y.reshape(-1, 1)

    root = HLCA(lca, mog, Y, 0)

    expand(root, 2, 24)

    for pre, _, node in RenderTree(root):
        print(pre, end='')
        print("{} {} {}".format(node.upperClass, len(node.data), node.ratio))
