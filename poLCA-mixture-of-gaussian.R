#!/usr/bin/env RScript

library(poLCA)


mof = read.csv('mixture-of-gaussian.csv',header=FALSE)

mof$V1 = as.integer(mof$V1) + 5
mof$V2 = as.integer(mof$V2) + 5

f = cbind(V1, V2) ~ 1

lca2 = poLCA(f, mof, nclass=2, nrep = 5)

lca3 = poLCA(f, mof, nclass=3, nrep = 5)

lca4 = poLCA(f, mof, nclass=4, nrep = 5)
