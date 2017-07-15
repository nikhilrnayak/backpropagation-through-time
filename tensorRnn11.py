# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:59:39 2017

@author: Nikhil Nayak
"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
df_tri = pd.read_csv('triang.csv')
df_sin = pd.read_csv('sin.csv')

X = df_tri.as_matrix()
Y = df_sin.as_matrix()

#define constants
batch_size = 1000
num_inps = 2
num_hidden = 15
num_op = 2
mu = 0.3
epoch_len = 1000
epochs = 100


#left rotate for output sequence
X_ = np.roll(X, -1, 0)
Y_ = np.roll(Y, -1, 0)


inp_mat = np.vstack([X.T, Y.T]).T
print(inp_mat.shape)

opt_mat = np.vstack([X_.T, Y_.T]).T
print(opt_mat.shape)

#inp_mat_1 = tf.pack(tf.transpose(inp_mat, perm=[1, 0 , 2]))
inp_mat = np.reshape(inp_mat, [1, 1000, 2])
opt_mat = np.reshape(opt_mat, [1000, 2])
store_op = np.zeros([1000, 2], dtype = np.float32)
store_op_1 = np.zeros([1000, 2], dtype = np.float32)
store_op_2 = np.zeros([1000, 2], dtype = np.float32)
cost_store = np.zeros([100,1])
cost_store_1 = np.zeros([200, 1])
cost_store_2 = np.zeros([100, 1])

#print(inp_mat.shape)

#define the graph

tfX = tf.placeholder(dtype = tf.float32, shape=[None, None, 2])
tfY = tf.placeholder(dtype = tf.float32, shape=[None, 2])
w0 = np.random.randn(num_hidden, num_op)/np.sqrt(num_hidden + num_op)
#w0 = np.random.randn(2, num_hidden, 1000)/np.sqrt(num_hidden + num_op)
b0 = np.zeros([num_op])
w0 = tf.Variable(w0.astype(np.float32))
b0 = tf.Variable(b0.astype(np.float32))
rnn_Cell = tf.contrib.rnn.BasicRNNCell(num_units = 15, activation = tf.tanh)
prev_state = rnn_Cell.zero_state(1, dtype = tf.float32)
#prev_state = tf.Variable(tf.float32, [1000, 2, 1, rnn_cell.state_size])
output = np.zeros([1, 15], dtype = np.float32)
output, prev_state = tf.nn.dynamic_rnn( cell = rnn_Cell, dtype = tf.float32, inputs = tfX, initial_state = prev_state)

#output = tf.transpose(output, perm = [2, 0, 1])
#output = tf.reshape(output, [1000, 15])
logits = tf.matmul(output[0], w0) + b0
cost_fn = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tfY, tf.nn.tanh(logits)))))
#train_fn = tf.train.AdamOptimizer(learning_rate = mu).minimize(cost_fn)
train_fn = tf.train.GradientDescentOptimizer(learning_rate = mu).minimize(cost_fn)
print(tf.shape(output))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for ind_i in range (0, epochs):
    res = sess.run([train_fn, cost_fn, logits], feed_dict={tfX: inp_mat, tfY: opt_mat})
    cost = res[1]
    store_op = res[2]
    print(cost)
    cost_store[ind_i, :] = cost
plt.plot(store_op[:, 0])
plt.show()
plt.plot(store_op[:, 1])
plt.show()
plt.plot(store_op[:, 0], store_op[:, 1])
plt.show()
plt.plot(cost_store)
plt.show()
store_ip = np.reshape(store_op, [1, 1000, 2])
for ind_i in range (0, 200):
    res_1 = sess.run([train_fn, cost_fn, logits], feed_dict={tfX: store_ip, tfY: opt_mat})
    cost_1 = res_1[1]
    store_op_1 = res_1[2]
    store_ip = np.roll(np.reshape(store_op, [1, 1000, 2]), 1, 0)
    #store_op = np.roll(store_op, -1, 1)
    print(cost_1)
    cost_store_1[ind_i, :] = cost_1
plt.plot(store_op_1[:, 0])
plt.show()
plt.plot(store_op_1[:, 1])
plt.show()
plt.plot(store_op_1[:, 0], store_op_1[:, 1])
plt.show()
plt.plot(cost_store_1)
plt.show()
store_ip_2 = np.reshape(store_op_1, [1, 1000, 2])
for ind_i in range (0, 1):
    res_2 = sess.run([cost_fn, logits], feed_dict={tfX: store_ip_2, tfY: opt_mat})
    cost_2 = res_2[0]
    store_op_2 = res_2[1]
    store_ip_2 = np.roll(np.reshape(store_op_2, [1, 1000, 2]), 1, 0)
    print(cost_2)
    cost_store_2[ind_i, :] = cost_2
plt.plot(store_op_2[:, 0])
plt.show()
plt.plot(store_op_2[:, 1])
plt.show()
plt.plot(store_op_2[:, 0], store_op_2[:, 1])
plt.show()
plt.plot(cost_store_2)
plt.show()
sess.close()