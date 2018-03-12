#!/usr/bin/env python


from __future__ import print_function
from hierachicalLCA import HLCA, expand, predict, predictOverTree, expandWithoutRatio, predictWithoutRatio
from lca import LCA
from anytree import RenderTree
import pandas as pd
import numpy as np


def sampling(opred, Y):
    samples = np.random.binomial(1, opred)
    return sum(samples == Y) / float(len(Y))
    

if __name__ == '__main__':
    # filters
    Sex = 2
    Children = 0

    tt = pd.read_csv('train_with_familyrole.csv')
    tt['Age'] = pd.cut(tt.Age, bins=[-np.inf, 3, 16, 25, 48, 65, 80, np.inf], labels=range(7))
    tt['Fare'] = pd.cut(tt.Fare, bins=[-np.inf, 4, 8, 15, 30, 100, np.inf], labels=range(6))

    tt0 = tt[(tt.Survived == 0)]
    tt1 = tt[(tt.Survived == 1)]

    keeps = ["Age", "Embarked", "Entourage", "Fare", "Parch", "Pclass", "Sex", "SibSp", "FamilySize", "ischild", "isfather", "ismother", "ishusband", "iswife", "isentourage", "aliveChildren", "aliveFather", "aliveMother", "aliveHusband", "aliveWife", "aliveEntourage"]

    data = tt[keeps].values
    data0 = tt0[keeps].values
    data1 = tt1[keeps].values
    Y = tt["Survived"].values

    # print(np.max(data, axis=0))

    N_VALUES = [7, 4, 8, 6, 7, 4, 3, 9, 12, 2, 2, 2, 2, 2, 2, 4, 3, 5, 3, 3, 5]
    lca = LCA(max_nclass=9, n_values=N_VALUES)
    lca.fit(data0)
    root0 = HLCA(lca, data0, None, 0, 0.35)

    expandWithoutRatio(root0, 5, N_VALUES, 9, max_level=3)

    lca = LCA(max_nclass=2, n_values=N_VALUES)
    lca.fit(data1)
    root1 = HLCA(lca, data1, None, 0, 0.35)

    expandWithoutRatio(root1, 5, N_VALUES, 9, max_level=3)

    for pre, _, node in RenderTree(root0):
        print(pre, end='')
        print("{} {} {}".format(node.upperClass, len(node.data), [len(sd) for sd in node.subdata]))

    for pre, _, node in RenderTree(root1):
        print(pre, end='')
        print("{} {} {}".format(node.upperClass, len(node.data), [len(sd) for sd in node.subdata]))

    pred = predictWithoutRatio(data, root0, root1)
    print("train accurate: ")
    print(sum(pred == tt.Survived.values) / float(len(data)))


    test = pd.read_csv('test_with_familyrole.csv')
    # test = test[(test.Sex == Sex) & (test.ischild == Children)]

    test['Age'] = pd.cut(test.Age, bins=[-np.inf, 16, 25, 48, 65, 80, np.inf], labels=range(6))
    test['Fare'] = pd.cut(test.Fare, bins=[-np.inf, 8, 15, 30, 100, np.inf], labels=range(5))

    testdata = test[keeps].values
    testY = test["Survived"].values

    testpred = predictWithoutRatio(testdata, root0, root1)
    print("test accurate: ")
    print(sum(testpred == testY) / float(len(testY)))
