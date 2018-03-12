#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from tabulate import tabulate


X = np.array(range(10), dtype=np.float32)
Y = X * 3. + 7. + np.random.normal(scale=0.5, size=X.shape)

W = tf.Variable([1.], tf.float32)
b = tf.Variable([5.], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for i in range(10000):
    sess.run(train, {x:X, y:Y})

print(sess.run([W, b, loss], {x:X, y:Y}))
est = sess.run(linear_model, {x:X})

table = [X, Y, est]
print(tabulate(table))
