#! /usr/bin/env python

# BRML Example 11.2

import tensorflow as tf
import numpy as np
from tabulate import tabulate

v = tf.constant(2.75)
theta = tf.Variable(2.95)
qh1 = tf.Variable(0.5)
qh2 = tf.Variable(0.5)

# E-step

qh1_s = tf.exp(-(v - theta) ** 2) / (tf.exp(-(v - theta) ** 2) + tf.exp(-(v - 2 * theta) ** 2))

# M-step

theta_s = v * (qh1 * 1. + qh2 * 2.) / (qh1 * 1. ** 2 + qh2 * 2. ** 2)

# Conditional Probability

prop = (tf.exp(-(v - theta) ** 2) + tf.exp(-(v - 2 * theta) ** 2)) / (2 * tf.sqrt(3.1415926))


# run
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    # E-step
    new_qh1 = sess.run(qh1_s)
    # print("new_qh1 = {}".format(new_qh1))
    fixqh1 = tf.assign(qh1, new_qh1)
    fixqh2 = tf.assign(qh2, 1. - new_qh1)
    sess.run([fixqh1, fixqh2])
    # M-step
    new_theta = sess.run(theta_s)
    # print("new_theta = {}".format(new_theta))
    fixtheta = tf.assign(theta, new_theta)
    sess.run([fixtheta])
    if i % 5 == 0:
        # output
        vars = sess.run([prop, qh1, qh2, theta])
        print(vars)

