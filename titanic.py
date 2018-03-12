#!/usr/bin/env python


from __future__ import print_function
from hierachicalLCA import HLCA, expand, predict, predictOverTree
from lca import LCA
from anytree import RenderTree
import pandas as pd
import numpy as np


def sampling(opred, Y):
    samples = np.random.binomial(1, opred)
    return sum(samples == Y) / float(len(Y))
    

if __name__ == '__main__':
    import sys
    # filters
    # female
    if sys.argv[1] == 'female':
        Sex = 1
        Children = 0
        min_split = 9
        max_level = 2
        root_nclass = 9
        node_nclass = 3

    # male
    if sys.argv[1] == 'male':
        Sex = 2
        Children = 0
        min_split = 5
        max_level = 4
        root_nclass = 9
        node_nclass = 3

    # children
    if sys.argv[1] == 'children':
        Children = 1
        min_split = 9
        max_level = 2
        root_nclass = 9
        node_nclass = 3

    # single
    if sys.argv[1] == 'single':
        Sex = 2
        min_split = 9
        max_level = 2
        root_nclass = 9
        node_nclass = 3

    # keeps = ["Age", "Embarked", "Entourage", "Fare", "Parch", "Pclass", "Sex", "SibSp", "FamilySize", "ischild", "isfather", "ismother", "ishusband", "iswife", "isentourage", "numchildren", "hasfather", "hasmother", "hashusband", "haswife", "numentourage", "aliveChildren", "aliveFather", "aliveMother", "aliveHusband", "aliveWife", "aliveEntourage"]
    # for female
    # keeps = ["Age", "Embarked", "Entourage", "Fare", "Parch", "Pclass", "Sex", "SibSp", "FamilySize", "ischild", "isfather", "ismother", "ishusband", "iswife", "isentourage", "aliveChildren", "aliveFather", "aliveMother", "aliveHusband", "aliveWife", "aliveEntourage"]
    # for male
    keeps = ["Age", "hascabin", "Fare", "Pclass", "Sex", "ischild", "isfather", "ismother", "ishusband", "iswife", "isentourage", "numchildren", "hasfather", "hasmother", "hashusband", "haswife", "numentourage", "aliveChildren", "aliveFather", "aliveMother", "aliveHusband", "aliveWife", "aliveEntourage"]

    tt = pd.read_csv('train_with_familyrole.csv')
    test = pd.read_csv('test_with_familyrole.csv')
    # FIXME, remove the following filter
    if sys.argv[1] == 'female' or sys.argv[1] == 'male':
        tt = tt[(tt.Sex == Sex) & (tt.ischild == Children) & (tt.FamilySize > 1)]
        test = test[(test.Sex == Sex) & (test.ischild == Children) & (test.FamilySize > 1)]
    elif sys.argv[1] == 'children':
        tt = tt[(tt.ischild == Children)]
        test = test[(test.ischild == Children)]
    elif sys.argv[1] == 'single':
        tt = pd.read_csv('train_with_single.csv')
        test = pd.read_csv('test_with_single.csv')
        tt = tt[(tt.FamilySize == 1)]
        test = test[(test.FamilySize == 1)]
    else:
        print("Invalid argument: {}".format(sys.argv[1]))
        sys.exit(1)


    tt['Age'] = pd.cut(tt.Age, bins=[-np.inf, 3, 16, 25, 48, 65, 80, np.inf], labels=range(7))
    tt['Fare'] = pd.cut(tt.Fare, bins=[-np.inf, 4, 8, 15, 30, 100, np.inf], labels=range(6))
    tt['hascabin'] = tt.CabinNum.apply(lambda c: 1 if c > 0 else 0)

    data = tt[keeps]
    Y = tt["Survived"]

    # print(np.max(data, axis=0))

    # N_VALUES = [7, 4, 8, 6, 7, 4, 3, 9, 12, 2, 2, 2, 2, 2, 2, 10, 2, 2, 2, 2, 9, 4, 3, 5, 3, 3, 6]
    # for female
    # N_VALUES = [7, 4, 8, 6, 7, 4, 3, 9, 12, 2, 2, 2, 2, 2, 2, 4, 3, 5, 3, 3, 6]
    # for male
    N_VALUES = [7, 2, 6, 4, 3, 2, 2, 2, 2, 2, 2, 10, 2, 2, 2, 2, 9, 4, 3, 5, 3, 3, 6]
    lca = LCA(max_nclass=root_nclass, n_values=N_VALUES)
    lca.fit(data)
    root = HLCA(lca, data, Y, 0, 0.35)

    expand(root, node_nclass, N_VALUES, 0.89, min_split, max_level=max_level)

    for pre, _, node in RenderTree(root):
        print(pre, end='')
        print("{} {} {} {}".format(node.upperClass, len(node.data), node.ratio, [len(sd) for sd in node.subdata]))

    opred = predict(data, root)
    pred = np.round(opred)
    print(sum(pred == tt.Survived.values) / float(len(data)))

    print("train accurate: ")
    for _ in range(5):
        print(sampling(opred, Y))

    test['Age'] = pd.cut(test.Age, bins=[-np.inf, 16, 25, 48, 65, 80, np.inf], labels=range(6))
    test['Fare'] = pd.cut(test.Fare, bins=[-np.inf, 8, 15, 30, 100, np.inf], labels=range(5))
    test['hascabin'] = test.CabinNum.apply(lambda c: 1 if c > 0 else 0)

    testdata = test[keeps]
    testY = test["Survived"]

    testopred = predict(testdata, root)
    testpred = np.round(testopred)
    print("Round accurate: ")
    print(sum(testpred == testY) / float(len(testY)))
    print("test accurate: ")
    for _ in range(5):
        print(sampling(testopred, testY))


    treeopred = predictOverTree(testdata, root)
    treepred = np.round(treeopred[:, 0])
    print("Tree round accurate: ")
    print(sum(treepred == testY) / float(len(testY)))
    print("Tree test accurate: ")
    for _ in range(5):
        print(sampling(treeopred[:, 0], testY))
