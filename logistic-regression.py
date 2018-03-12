#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from tabulate import tabulate

# X = np.array([1,2,1,2,1,9,8,9,8,9], dtype=np.float32)
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
Y = np.array([0,0,0,0,0,1,1,1,1,1], dtype=np.float32)

W = tf.Variable([.1], tf.float32)
b = tf.Variable([.1], tf.float32)
x = tf.placeholder(tf.float32)

logistic_model = tf.sigmoid(W * x + b)

y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(- (1 - y) * tf.log(1 - logistic_model) - y * tf.log(logistic_model))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for i in range(9000):
    sess.run(train, {x:X, y:Y})

print(sess.run([W, b, loss], {x:X, y:Y}))
est = sess.run(logistic_model, {x:X})

table = [X, Y, est]
print(tabulate(table))
