# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:57:52 2017

@author: Nikhil Nayak

remove momentum or decrease it futher to improve results
set the non linear derivatives to 1 to get better results

batch_size = 1000
num_inps = 2
num_hidden = 15
num_ops = 2
mu = 0.1
epoch_len = 1000
epochs = 300
time_batch_size = 10
num_batches = epoch_len//time_batch_size
mu = 0.0002

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df_tri = pd.read_csv('triang.csv')
df_sin = pd.read_csv('sin.csv')

X = df_tri.as_matrix()
Y = df_sin.as_matrix()
Z = np.ones([1000, 1])

#define constants
batch_size = 1000
num_inps = 2
num_hidden = 15
num_ops = 2
mu = 0.1
epoch_len = 1000
epochs = 200
time_batch_size = 50
num_batches = epoch_len//time_batch_size
mu = 0.003

#left rotate for output sequence
X_ = np.roll(X, -1, 0)
Y_ = np.roll(Y, -1, 0)


inp_mat = np.vstack([X.T, Y.T, Z.T]).T
print(inp_mat.shape)

#define the nonlinear activation functions
def tanh(inp):
    e_x = np.exp(inp)
    e_x_ = np.exp(-inp)
    return np.divide((e_x - e_x_) , (e_x + e_x_))

def tanh_1(inp):
    return (1 - np.multiply(inp, inp))

opt_mat = np.vstack([X_.T, Y_.T]).T
print(opt_mat.shape)

store_op = np.array([[1000, 2]], dtype = np.float32)

store_op_2 = np.zeros([1000, 2], dtype = np.float32)

#define weight matrices for the different layers
U = np.random.rand(num_hidden, num_inps + 1)
W = np.random.rand(num_hidden, num_hidden)
V = np.random.rand(num_ops, num_hidden + 1)

#define the intermediate result variables
#h_in = np.zeros([num_hidden + 1, time_batch_size + 1], dtype = np.float32)
#h_in[num_hidden, :] = np.ones([1, time_batch_size + 1], dtype = np.float32)
y_in = np.zeros([num_ops, epoch_len + 1])

cost = np.zeros([1, 1000])
cost_epoch = np.zeros([epochs, 1], dtype = np.float32)

#define the graph
for ind_e in range(0, epochs):
     for ind_i in range(0, num_batches): #num_batches
        h_in = np.zeros([num_hidden + 1, time_batch_size + 1], dtype = np.float32)
        h_in[num_hidden, :] = np.ones([1, time_batch_size + 1], dtype = np.float32)
        err_x = np.zeros([1, time_batch_size + 1], dtype=np.float32)
        err_y = np.zeros([1, time_batch_size + 1], dtype=np.float32)
        dy = np.zeros([time_batch_size + 1, num_ops, num_hidden + 1], dtype=np.float32)
        grad_V = np.zeros([num_ops, num_hidden + 1], dtype=np.float32)
        chain = np.eye(num_hidden, dtype = np.float32)
        store_chain_W = np.zeros([time_batch_size + 1, num_hidden, num_hidden], dtype = np.float32)
        store_chain_U = np.zeros([time_batch_size + 1, num_hidden, num_inps + 1], dtype = np.float32)
        dwhh = np.zeros([time_batch_size + 1, num_hidden, num_hidden], dtype = np.float32)
        dwxh = np.zeros([time_batch_size + 1, num_hidden, num_inps + 1], dtype = np.float32)
        grad_W = np.zeros([num_hidden, num_hidden], dtype = np.float32)
        grad_U = np.zeros([num_hidden, num_inps + 1], dtype = np.float32)
        grad_VM = np.zeros([num_ops, num_hidden + 1], dtype=np.float32)
        grad_WM = np.zeros([num_hidden, num_hidden], dtype = np.float32)
        grad_UM = np.zeros([num_hidden, num_inps + 1], dtype = np.float32)
        cost_check = 0
        for ind_j in range(0, time_batch_size):#time_batch_size):
            #feedforward part
            #input to hidden
            U_X = np.dot(U, inp_mat[(ind_i * time_batch_size) + ind_j, :])
            #hidden to hidden
            W_H = np.dot(W, h_in[0:num_hidden, ind_j])
            #non linear activation of hidden
            h_in[0:num_hidden, ind_j + 1] = np.tanh(U_X + W_H)
            #set bias inputs to 1
            h_in[num_hidden, ind_j + 1] = 1
            #hidden to output
            V_H = np.dot(V, h_in[:, ind_j + 1])
            y_in[:, (ind_i * time_batch_size) + ind_j + 1] = np.tanh(V_H)
            #compute the cost
            err_x[0, ind_j + 1] = opt_mat[(ind_i * time_batch_size) + ind_j, 0] - y_in[0, (ind_i * time_batch_size) + ind_j + 1]
            err_y[0, ind_j + 1] = opt_mat[(ind_i * time_batch_size) + ind_j, 1] - y_in[1, (ind_i * time_batch_size) + ind_j + 1]
            cost[0, (ind_i * time_batch_size) + ind_j] =  np.square(err_x[0, ind_j + 1]) + np.square(err_y[0, ind_j + 1])
            cost_check += cost[0, (ind_i * time_batch_size) + ind_j]
            #compute the gradients
            dfy_x = 1#tanh_1(y_in[0, (ind_i * time_batch_size) + ind_j + 1])
            dfy_y = 1#tanh_1(y_in[1, (ind_i * time_batch_size) + ind_j + 1])
            dy[ind_j + 1, 0, :] = np.multiply(np.multiply(err_x[0, ind_j + 1], dfy_x), h_in[0:num_hidden + 1, ind_j + 1].T)
            dy[ind_j + 1, 1, :] = np.multiply(np.multiply(err_y[0, ind_j + 1], dfy_y), h_in[0:num_hidden + 1, ind_j + 1].T)
            grad_V += dy[ind_j + 1, :, :]
            #print(err_x[0, ind_j + 1], dfy_x)
            temp = np.multiply(np.multiply(err_x[0, ind_j + 1], dfy_x), V[0,:]) + np.multiply(np.multiply(err_y[0, ind_j + 1], dfy_y), V[1, :])
            dh = np.ones([15, 1], dtype = np.float32)#tanh_1(h_in[:, ind_j + 1])
            dh = np.multiply(0.09, dh)
            chain = np.dot(np.diag(dh[0:num_hidden, 0]), W)
            #dh = tanh_1(h_in[0:num_hidden, ind_j + 1])
            #chain = np.dot(np.diag(dh[0:num_hidden]), W)
            #chain = np.dot(W.T, np.diag(dh[0:num_hidden, 0]))
            #chain = np.dot(W, np.diag(dh[0:num_hidden]))
            store_chain_W[ind_j + 1, :, :] = np.diag(h_in[0:num_hidden, ind_j])
            store_chain_U[ind_j + 1, :, :] = np.tile(inp_mat[(ind_i * time_batch_size) + ind_j, :].T, (num_hidden, 1))
            for ind_k in range (0, ind_j + 1):
                store_chain_W[ind_k + 1, :, :] = np.dot(chain, store_chain_W[ind_k + 1, :, :])
                store_chain_U[ind_k + 1, :, :] = np.dot(chain, store_chain_U[ind_k + 1, :, :])
                dwhh[ind_k + 1, :, :] += np.dot(np.diag(temp[0:num_hidden]), store_chain_W[ind_k + 1, :, :])
                dwxh[ind_k + 1, :, :] += np.dot(np.diag(temp[0:num_hidden]), store_chain_U[ind_k + 1, :, :])
        for ind_k in range (0, time_batch_size):
            grad_W += dwhh[ind_k + 1, :, :]
            grad_U += dwxh[ind_k + 1, :, :]
        for ind_x in [grad_V, grad_W, grad_U]:
            np.clip(ind_x, -5, 5, out = ind_x)
        grad_VM = mu * grad_V + 0.05 * grad_VM
        V = V + grad_VM
        grad_WM = mu * grad_W + 0.05 * grad_WM
        W = W + grad_WM
        grad_UM = mu * grad_U + 0.05 * grad_UM
        U = U + grad_UM
        h_in[:, 0] = h_in[:, time_batch_size]
        #print(cost_epoch)
        cost_epoch[ind_e, :] += cost_check
     cost_epoch[ind_e, :] /= 1000
     print(cost_epoch[ind_e, :])
plt.plot(y_in[0, :])
plt.show()
plt.plot(y_in[1, :])
plt.show()
plt.plot(y_in[0, :], y_in[1, :])
plt.show()
plt.plot(cost_epoch[:, :])
plt.show()

inp_check = np.zeros([3, 1], dtype = np.float32)
inp_check = inp_mat[0, :]
h_prev = np.zeros([num_hidden + 1], dtype = np.float32)
h_prev[num_hidden] = 1
h_now = np.zeros([num_hidden + 1], dtype = np.float32)
h_now[num_hidden] = 1
cost_store_2 = np.zeros([1000], dtype = np.float32)
for ind_i in range (0, 1000):
    U_X = np.dot(U, inp_check)
    #hidden to hidden
    W_H = np.dot(W, h_prev[0:num_hidden])
    #non linear activation of hidden
    h_now[0:num_hidden] = np.tanh(U_X + W_H)
    #set bias inputs to 1
    h_now[num_hidden] = 1
    #hidden to output
    V_H = np.dot(V, h_now[:])
    y_check = np.tanh(V_H)
    store_op_2[ind_i, :] = y_check
    #compute the cost
    err_x_check = opt_mat[ind_i, 0] - y_check[0]
    err_y_check = opt_mat[ind_i, 1] - y_check[1]
    cost_store_2[ind_i] =  np.square(err_x_check) + np.square(err_y_check)
    h_prev = h_now
    inp_check[0:2] = y_check
    
plt.plot(store_op_2[:, 0], store_op_2[:, 1])
plt.show()