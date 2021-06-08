# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Activation, LSTM, GRU, Input, Lambda, Concatenate, Conv1D, LayerNormalization
import keras.backend as K
import tensorflow as tf

sys.path.append(os.path.abspath('../'))
from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
from utils.keras_helpers import *
#from ops.complex_ops import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.set_printoptions(precision=3, threshold=3, edgeitems=3)





#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class tdnaec(object):

    def __init__(self, nbatch=25):

        self.name = os.path.splitext(os.path.basename(sys.argv[0]))[0]               # filename of this script without extension
        self.file_date = os.path.getmtime(sys.argv[0])                               # timestamp of this script
        self.weights_file = '../weights/' + self.name + '_weights.h5'
        self.predictions_file = '../predictions/' + self.name + '.mat'
        self.logfile = self.name+'_logfile.txt'

        self.fgen = feature_generator()
        self.logger = Logger(self.name)
        self.samples = self.fgen.samples
        self.silence = self.fgen.silence
        self.iterations = 100000
        self.nbatch = nbatch

        self.fs = self.fgen.fs
        self.wlen_x = int(self.fs*0.100)
        self.wlen_z = 256
        self.shift = self.wlen_z//2
        self.nbin = 200

        self.create_model()



    #---------------------------------------------------------
    def forward_long(self, x):

        nbatch = tf.shape(x)[0]
        pad = tf.zeros((nbatch, self.wlen_x-self.shift), dtype=tf.float32)
        x = tf.concat([pad, x], axis=1)
        Bx = tf.signal.frame(x, self.wlen_x, self.shift, axis=-1)            # shape = (nbatch, nfram, wlen_x)

        return Bx



    #---------------------------------------------------------
    def forward(self, x):

        nbatch = tf.shape(x)[0]
        pad = tf.zeros((nbatch, self.wlen_z-self.shift), dtype=tf.float32)
        x = tf.concat([pad, x], axis=1)
        Bx = tf.signal.frame(x, self.wlen_z, self.shift, axis=-1)            # shape = (nbatch, nfram, wlen_x)

        return Bx



    #---------------------------------------------------------
    def inverse(self, Bx):

        x = tf.signal.overlap_and_add(Bx, self.shift)                                # shape = (nbatch, samples)
        x = x[:,(self.wlen_z-self.shift):]

        return x



    #---------------------------------------------------------
    def normalize(self, X):

        den = tf.math.sqrt(tf.reduce_sum(X**2, axis=-1, keepdims=True)) + 1e-3
        X /= den

        return X



    #---------------------------------------------------------
    def cost(self, inp):
        
        z = inp[0]                                    # shape = (nbatch, samples)
        s = inp[1]                                    # shape = (nbatch, samples)
        e = inp[2]                                    # shape = (nbatch, samples)

        nbatch = tf.shape(s)[0]
        samples = tf.shape(s)[1]

        pad_z = tf.zeros((nbatch, samples-tf.shape(z)[1]), dtype=tf.float32)
        z = tf.concat([z, pad_z], axis=1)
        z -= tf.reduce_mean(z, axis=-1, keepdims=True)

        Ps = tf.reduce_mean(s*s, axis=-1) + 1e-6
        Pn = tf.reduce_mean((s-z)**2, axis=-1) + 1e-6
        sdr = 10*log10(Ps) - 10*log10(Pn)

        Pe = tf.reduce_mean(e*e, axis=-1) + 1e-6
        Pz = tf.reduce_mean(z*z, axis=-1) + 1e-6
        erle = 10*log10(Pe) - 10*log10(Pz)
        erle = tf.minimum(erle, 30)

        Ps = tf.reduce_sum(s*s, axis=-1)
        idx = tf.cast(Ps > 1e-6, tf.float32)
        cost = -tf.reduce_mean(sdr*idx + (1-idx)*erle)

        return [cost,z]



    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        x = Input(batch_shape=(self.nbatch, None), dtype=tf.float32)                          # shape = (nbatch, samples)
        y = Input(batch_shape=(self.nbatch, None), dtype=tf.float32)                          # shape = (nbatch, samples)
        d = Input(batch_shape=(self.nbatch, None), dtype=tf.float32)                          # shape = (nbatch, samples)
        e = Input(batch_shape=(self.nbatch, None), dtype=tf.float32)                          # shape = (nbatch, samples)
        s = Input(batch_shape=(self.nbatch, None), dtype=tf.float32)                          # shape = (nbatch, samples)

        Bx = Lambda(self.forward_long)(x)
        By = Lambda(self.forward_long)(y)
        Bd = Lambda(self.forward_long)(d)
        Be = Lambda(self.forward_long)(e)

        X = Dense(units=self.nbin, activation='linear')(Bx)
        Y = Dense(units=self.nbin, activation='linear')(By)
        D = Dense(units=self.nbin, activation='linear')(Bd)
        E = Dense(units=self.nbin, activation='linear')(Be)
        Px = LayerNormalization()(X**2)
        Py = LayerNormalization()(Y**2)
        Pd = LayerNormalization()(D**2)
        Pe = LayerNormalization()(E**2)
        P = Concatenate(axis=-1)([Px,Py,Pd,Pe])

        P = Dense(units=self.nbin, activation='tanh')(P)
        P = Dense(units=self.nbin, activation='tanh')(P)
        P = GRU(units=self.nbin, activation='tanh', return_sequences=True, stateful=True)(P)
        P = Dense(units=self.nbin, activation='softplus')(P)

        Be = Lambda(self.forward)(e)
        E = Dense(units=self.nbin, activation='linear')(Be)
        E = GRU(units=self.nbin, activation='tanh', return_sequences=True, stateful=True)(E)
        Z = E*P
        Z = Dense(units=self.wlen_z, activation='linear')(Z)
        z = Lambda(self.inverse)(Z)

        cost,z = Lambda(self.cost)([z,s,e])


        self.model = Model(inputs=[x,y,d,e,s], outputs=[z,P])
        self.model.add_loss(cost)
        self.model.compile(loss=None, optimizer='adam')

        print(self.model.summary())
        
        
        try:
            self.model.load_weights(self.weights_file)
        except:
            print('error loading weights file: %s' % self.weights_file)



    #---------------------------------------------------------
    def save_prediction(self,):

        x,y,d,e,s = self.fgen.load_train(self.nbatch, scenario='doubletalk')
        #x,y,d,e,s = self.fgen.load_train(self.nbatch)

        z,P = self.model.predict([x,y,d,e,s], batch_size=self.nbatch)

        data = {
                'x': x[0,...],
                'y': y[0,...],
                'd': d[0,...],
                'e': e[0,...],
                's': s[0,...],
                'z': z[0,...],
                'P': P[0,...],
               }
        save_numpy_to_mat(self.predictions_file, data)
        
        if np.isnan(np.sum(z)):
            self.model.load_weights(self.weights_file)
        else:
            self.model.save_weights(self.weights_file)



    #---------------------------------------------------------
    def train_model(self):

        print('train the model')
        i = 0
        while (i<self.iterations) and (self.file_date == os.path.getmtime(sys.argv[0])):

            #x,y,d,e,s = self.fgen.load_train(self.nbatch, scenario='doubletalk')
            x,y,d,e,s = self.fgen.load_train(self.nbatch)
            self.model.fit([x,y,d,e,s], None, batch_size=self.nbatch, epochs=1, verbose=0, callbacks=[self.logger])

            i += 1
            if (i%20)==0:
                self.save_prediction()



    #---------------------------------------------------------
    def test_model(self,):

        print('test the model')
        '''
        for i in range(self.fgen.test_set_length):
            x,y,d,e,s = self.fgen.load_test(i, self.nbatch)
            z,P = self.model.predict([x,y,d,e,s])
            self.fgen.write_enhanced(z, i, self.name)
            print(self.name, ', writing enhanced file:', i, '/', self.fgen.test_set_length)
        '''
        for i in range(self.fgen.blind_test_set_length):
            x,y,d,e,s = self.fgen.load_test_blind(i, self.nbatch)
            z,P = self.model.predict([x,y,d,e,s])
            self.fgen.write_enhanced_blind(z, i, self.name)
            print(self.name, ', writing blind enhanced file:', i, '/', self.fgen.blind_test_set_length)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='time-domain neural echo controller')
    parser.add_argument('mode', help='mode: [train, test]', nargs='?', choices=('train', 'test'), default='train')
    args = parser.parse_args()


    if args.mode == 'train':
        dnn = tdnaec(nbatch=40)
        dnn.train_model()

    if args.mode == 'test':
        dnn = tdnaec(nbatch=1)
        dnn.test_model()


