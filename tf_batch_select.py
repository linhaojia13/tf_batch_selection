#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:04:30 2019

@author: ubuntu
"""

import sys
import os
import time
import numpy as np
import tensorflow as tf
import math
import random
from bisect import bisect_right
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)
    
    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')
        
    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, Y_train = data[0]
    X_vali, Y_vali = data[1]
    X_test, Y_test = data[2]

    #reshape the image from (-1,784) to (-1, 28, 28, 1)
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_vali = X_vali.reshape((-1, 28, 28, 1))
    X_test = X_test.reshape((-1, 28, 28, 1))
    
    # The targets are int64, we cast them to int8 for GPU compatibility.
    Y_train = Y_train.astype(np.uint8)
    Y_vali = Y_vali.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)
    
    return X_train, Y_train, X_vali, Y_vali, X_test, Y_test 

def build_cnn(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        #First convolutional and pool layers
        w1 = tf.get_variable('w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0))
        net1 = tf.nn.conv2d(input=images, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
        net1 = net1 + b1
        net1 = tf.nn.relu(net1)
        net1 = tf.nn.avg_pool(net1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #Second convolutional and pool layers
        w2 = tf.get_variable('w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0))
        net2 = tf.nn.conv2d(input=net1, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
        net2 = net2 + b2
        net2 = tf.nn.relu(net2)
        net2 = tf.nn.avg_pool(net2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #firstly fully connected layer
        w3 = tf.get_variable('w3', [7 * 7 * 64, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [256], initializer=tf.constant_initializer(0))
        net3 = tf.reshape(net2, [-1, 7 * 7 * 64])
        net3 = tf.matmul(net3, w3)
        net3 = net3 + b3
        net3 = tf.nn.relu(net3)
        
        #second fully connected layer
        w4 = tf.get_variable('w4', [256, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [10], initializer=tf.constant_initializer(0))
        net4 = tf.matmul(net3, w4) + b4
        net4 = tf.nn.softmax(net4)
        
        return net4
        
def test(model='cnn', num_epochs=50, bs_begin=16, bs_end=16, fac_begin=100, fac_end=100, pp1 = 0, pp2 = 0, opt_type=1, nettype=1, adapt_type = 0, irun = 1):
    
#    # load the dataset
#    print("loading the dataset")
#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#    
#    X_train = mnist.train.images
#    Y_train = mnist.train.labels
#    print("X_train shape:", X_train.shape)
#    print("Y_train_shape:", Y_train.shape)
#    
#    X_validation = mnist.validation.images
#    Y_validation = mnist.validation.labels
#    print("X_validation shape:", X_validation.shape)
#    print("Y_validation shape:", Y_validation.shape)
#
#    X_test = mnist.test.images
#    Y_test= mnist.test.labels
#    print("X_test shape:", X_test.shape)
#    print("Y_test shape:", Y_test.shape)
#        
#    batch = mnist.train.next_batch(16)
#    print("image of one batch shape", batch[0].shape)
#    print("label of one batch shape", batch[1].shape)
#    
#    one_X_train = mnist.train.images[3]
#    print("one of X_train shape", one_X_train.shape)
#    one_Y_train = mnist.train.labels[3]
#    print("one of Y_train  ", one_Y_train)
#    plt.imshow(one_X_train.reshape(28, 28), cmap='Greys')
#    plt.show()

    print("loading the dataset")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train = mnist.train.images.reshape([-1, 28, 28, 1])
    Y_train = mnist.train.labels
    #print('X_train shape', X_train.shape)
    #print('Y_train_shape', Y_train.shape)

    x_placeholder = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, shape = [None, 10], name='y_placeholder')
    
    if model == 'cnn':
        y = build_cnn(x_placeholder)
    else:
        print('model error....')
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = y_placeholder*tf.log(y)
    loss_sum = -tf.reduce_sum(loss)
    loss = -tf.reduce_sum(loss, 1)
    
    if opt_type == 2:
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss_sum)
        opt_name = 'Adam'
    else:
        train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-06).minimize(loss_sum)
        opt_name = 'Adadelta'
    
    bfs = []
    ntraining = len(X_train)
    bfs = np.ndarray((ntraining, 2))
    l = 0
    for l in range(0, ntraining):
        bfs[l][0] = 1e+10
        bfs[l][1] = l
        #print('bfs[l][1]: ', bfs[l][1])
        #print('l:', l)
    
    prob = [None] * ntraining
    sumprob = [None] * ntraining
    
    filename = opt_name + "_{}_{}_{}_{}_{}".format(bs_end, pp1, pp2, fac_begin, fac_end) + ".txt"
    mult_bs = math.exp(math.log(bs_end/bs_begin)/num_epochs)
    mult_fac = math.exp(math.log(fac_end/fac_begin)/num_epochs) 
    
    sorting_evaluations_period = 100 #increase it if sorting is too expensive
    sorting_evaluations_ago = 2 * sorting_evaluations_period
    
    start_time0 = time.time()
    wasted_time = 0 # time wasted on computinf training and validation 
    start_train_time = time.time()
    cur_train_time = time.time()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # we iterate over epochs:
        for epoch in range(num_epochs):
            start_time = time.time()
            fac = fac_begin * math.pow(mult_fac, epoch)
            if(adapt_type == 0): #linear
                bs = bs_begin + (bs_end - bs_begin) * (float(epoch)/float(num_epochs - 1))
            if(adapt_type == 1): #exponential
                bs = bs_begin * math.pow(mult_bs, epoch)
            bs = int(math.floor(bs))
            
            if (fac == 1):
                print('************We are working on normal mode************')
                batch = mnist.train.next_batch(bs)
                inputs, targets = batch
                inputs = inputs.reshape([bs, 28, 28, 1])
                #train a batch
                _, loss_all, loss_val, ac = sess.run([train_step, loss, loss_sum, accuracy], 
                                   feed_dict = {x_placeholder:inputs, y_placeholder:targets})
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(loss_val), "{:.6f}".format(ac))
                cur_train_time = time.time()
                print("By now, we use time:", cur_train_time - start_train_time)
            else:
                print('************We are working on batch_selection mode************')
                mult = math.exp(math.log(fac)/ntraining)
                for i in range(ntraining):
                    if(i == 0): prob[i] = 1.0
                    else:       prob[i] = prob[i-1]/mult
                psum = sum(prob)
                prob = [v/psum for v in prob]
                for i in range(0, ntraining):
                    if (i == 0): sumprob[i] = prob[i]
                    else:        sumprob[i] = sumprob[i-1] + prob[i]
                
                stop = 0
                iter = 0
                while(stop == 0):
                    indexes = []
                    wrt_sorted = 1
                    if (sorting_evaluations_ago >= sorting_evaluations_period):
                        bfs = bfs[bfs[:,0].argsort()[::-1]]
                        sorting_evaluations_ago = 0
                    
                    stop1 = 0
                    while (stop1 == 0):
                        index = iter
                        if (wrt_sorted == 1):
                            randpos = min(random.random(), sumprob[-1])
                            index = bisect_right(sumprob, randpos) #o(;og(ntraining)), cheap
                        indexes.append(index)
                        iter = iter + 1
                        if(len(indexes) == bs) or (iter == len(X_train)):
                            stop1 = 1
                    sorting_evaluations_ago += bs
                    if (iter == len(X_train)):
                        stop = 1
                    
                    idxs = []
                    for idx in indexes:
                        idxs.append(bfs[idx][1])
                        #print('bfs[idx][1]: ', bfs[idx][1])
                    #print('idxs: ', idxs)
                    idxs = np.array(idxs)
                    idxs = idxs.astype(np.int8)
                    batch = X_train[idxs], Y_train[idxs]
                    inputs, targets = batch
                    
                #train a batch
                _, loss_all, loss_val, ac = sess.run([train_step, loss, loss_sum, accuracy], 
                                   feed_dict = {x_placeholder:inputs, y_placeholder:targets})
                cur_train_time = time.time()
                i = 0
                for idx in indexes:
                    #print('loss_all', loss_all[i])
                    bfs[idx][0] = loss_all[i]
                    i += 1
    
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(loss_val), "{:.6f}".format(ac))
                print("By now, we use time:", cur_train_time - start_train_time)
        saver.save(sess, 'model/model.ckpt')
        print('model has saved!')
        
                
            
    
def main():
    nettype = 2     # 1 - small network with 6 filters, 2 - bigger network with 32 filters, see function build_cnn for more details
    num_epochs = 500
    opt_type = 2
    adapt_type = 1
    #param 1
    irun = 1
    bs = 64
    bs_begin = bs
    bs_end = bs
    fac_begin = 1
    fac_end = 1
    pp1 = 0; pp2 = 0
    test('cnn', num_epochs, bs_begin, bs_end, fac_begin, fac_end, pp1, pp2, opt_type, nettype, adapt_type, irun)

#    #param 2
#    irun = 1
#    bs = 64
#    bs_begin = bs
#    bs_end = bs
#    fac_begin = 1e+8
#    fac_end = 1e+8
#    pp1 = 0; pp2 = 0
#    test('cnn', num_epochs, bs_begin, bs_end, fac_begin, fac_end, pp1, pp2, opt_type, nettype, adapt_type, irun)

    
    
    
    
    
if __name__ == '__main__':
    main()
