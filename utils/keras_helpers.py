# -*- coding: utf-8 -*-

__author__ = "Lukas Pfeifenberger"

import numpy as np
import sys
import time
import h5py

import keras.backend as K
import tensorflow as tf




#-----------------------------------------------------
def display_hd5_weights(hd5_file):

    data = h5py.File(hd5_file, 'r')

    print('_'*80)
    print('*** weights from file: ', hd5_file)

    for group1, member1 in data.items():
        for group2, member2 in member1.items():
            for group3, member3 in member2.items():
                print('layer_name =', group1, ', weight_name =', group3, ', shape =', member3.shape, 'mean value =', np.mean(member3.value))

    print('_'*80)



#-----------------------------------------------------
def display_model_weights(model):

    print('_'*80)
    print('*** weights from model: ', model.name)

    for layer in model.layers:
        for weight in layer.trainable_weights+layer.non_trainable_weights:

            print('layer_name =', layer.name, ', weight_name =', weight.name.split('/')[1], ', shape =', K.get_value(weight).shape, 'mean value =', np.mean(K.get_value(weight)))

    print('_'*80)



#-----------------------------------------------------
def load_hd5_layer(hd5_file, layer_name):

    data = h5py.File(hd5_file, 'r')

    if layer_name in data.keys():
        if len(data[layer_name]) > 0:
            layer = data[layer_name][layer_name]

            weights = {}
            for weight_name, weight_value in layer.items():
                weights[weight_name] = weight_value.value

            return weights

    return None



#-----------------------------------------------------
def sideload_weights(model, hd5_file, layer_name):

    hd5_weights = load_hd5_layer(hd5_file, layer_name)

    for layer in model.layers:

        if layer_name == layer.name:

            for weight in layer.trainable_weights+layer.non_trainable_weights:

                weight_name = weight.name.split('/')[1]
                if weight_name in hd5_weights.keys():

                    value = K.get_value(weight)
                    hd5_value = hd5_weights[weight_name]
                    if value.shape == hd5_value.shape:

                        K.set_value(weight, hd5_value)
                        print('*** updated ', layer_name, weight_name)





'''
#-----------------------------------------------------
# usage:   model.compile(loss=weighted_mse(Pz), optimizer='adam')
def weighted_mse(weight):

    def loss(y_true, y_pred):

        tmp = K.square(y_true-y_pred)
        return K.sum(tmp*weight) / K.sum(weight)

    return loss
'''


#-----------------------------------------------------
class Logger(tf.keras.callbacks.Callback):

    def __init__(self, name):
        self.name = name
        self.iteration = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
        self.losses = []
        self.erle = []
        self.sdr = []

    def on_batch_end(self, batch, logs=None):
        self.losses = np.append(self.losses, logs['loss'])
        #self.erle = np.append(self.erle, logs['erle'])
        #self.sdr = np.append(self.sdr, logs['sdr'])
        #print('end of batch: ', logs['loss'].shape)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = time.time()
        duration = self.epoch_time_end-self.epoch_time_start
        self.iteration += 1
        #print('end of epoch: ', self.losses.shape)

        #print('model: %s, iteration: %d, epoch: %d, runtime: %.3fs, loss: %.3f, ERLE: %.3fdB, SDR: %.3fdB' % \
        #    (self.name, self.iteration, epoch, duration, np.mean(self.losses), np.mean(self.erle), np.mean(self.sdr)) )

        print('model: %s, iteration: %d, epoch: %d, runtime: %.3fs, loss: %.3f' % \
            (self.name, self.iteration, epoch, duration, np.mean(self.losses)) )



#-----------------------------------------------------
def Debug(name, x):

    # print the dynamic shape of tensor x during runtime
    print_op = tf.print(name, '.shape =', tf.shape(x), '.dtype=', x.dtype, '.value=', x)
    with tf.control_dependencies([print_op]):
        return tf.identity(x)


#-----------------------------------------------------
def log10(x):

    return tf.math.log(x) / 2.302585092994046


#-----------------------------------------------------
def pow10(x):

    return tf.math.exp(x * 2.302585092994046)



#-----------------------------------------------------
# remove element at 'idx' and 'axis' from tensor x
def delete_element(x, idx, axis):

    mask = tf.one_hot(idx, tf.shape(x)[axis], on_value=0, off_value=1)
    y = tf.boolean_mask(x, mask, axis=axis)

    return y




#---------------------------------------------------------
def tensor_softmax(x, axis=-1):

    e_x = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    e_x_n = tf.reduce_sum(e_x, axis=axis, keepdims=True)
    return e_x / (e_x_n + 1e-6)



#---------------------------------------------------------
def batch_sum(x):

    all_but_first_axes = tuple(range(1, K.ndim(x)))
    return tf.reduce_sum(x, axis=all_but_first_axes)


#---------------------------------------------------------
def batch_mean(x):

    all_but_first_axes = tuple(range(1, K.ndim(x)))
    return tf.reduce_mean(x, axis=all_but_first_axes)


#---------------------------------------------------------
def weighted_mse(p_true, p_est, weight=None):

    mse = (p_true-p_est)**2

    if weight is None:
        return batch_mean(mse)
    else:
        return batch_sum(mse*weight) / batch_sum(weight)



#-----------------------------------------------------
def weighted_bce(p_true, p_est, weight=None):

    eps = 1e-6
    p_true = tf.clip_by_value(p_true, eps, 1.0-eps)
    p_est = tf.clip_by_value(p_est, eps, 1.0-eps)
    cce = p_true*tf.math.log(p_est) + (1-p_true)*tf.math.log(1-p_est)

    if weight is None:
        return -batch_mean(cce)
    else:
        return -batch_sum(cce*weight) / batch_sum(weight)


#-----------------------------------------------------
def weighted_cce(p_true, p_est, weight=None, axis=-1):

    p_true = tf.clip_by_value(p_true, 1e-6, 1.0)
    p_est = tf.clip_by_value(p_est, 1e-6, 1.0)
    cce = tf.reduce_sum(p_true*tf.log(p_est), axis=-1)

    if weight is None:
        return -batch_mean(cce)
    else:
        return -batch_sum(cce*weight) / batch_sum(weight)


