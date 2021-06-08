# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import *
from utils.mat_helpers import *




#--------------------------------------------------------------------------
# state-space AEC implementation based on:
# STATE-SPACE ARCHITECTURE OF THE PARTITIONED-BLOCK-BASED ACOUSTIC ECHO CONTROLLER
# Fabian Kuech, Edwin Mabande, Gerald Enzner, ICASSP 2014
#--------------------------------------------------------------------------

class ssaec_fast(object):

    #--------------------------------------------------------------------------
    def __init__(self, fs=16e3, wlen=512, tail_length=0.25): 

        self.fs = fs
        self.wlen = wlen
        self.tail_length = tail_length

        self.nbin = self.wlen//2+1
        self.shift = self.wlen//2
        self.nslices = int(self.tail_length*self.fs) // self.shift
        self.framerate = self.fs/self.shift

        self.window = create_hanning_window(self.wlen)

        self.A = 0.95                                                                # learning rate
        self.X = np.zeros((self.nbin, self.nslices), dtype=np.complex64)             # delay-line for the input x
        self.P = np.ones((self.nbin, self.nslices), dtype=np.float32)*10             # covariance matrix
        self.W = np.zeros((self.nbin, self.nslices), dtype=np.complex64)             # filter weights
        self.W2 = np.zeros((self.nbin, self.nslices), dtype=np.complex64)            # shadow filter weights

        # measure erle for shadow-weights
        self.Pd = 0
        self.Pe = 0
        self.Pe2 = 0
        self.alpha = 0.95



    #--------------------------------------------------------------------------
    def run(self, x=0, d=0, repeats=1):

        samples = x.shape[0]
        nfram = samples//self.shift

        # delay x by 20ms
        x = np.roll(x, int(-0.020*self.fs))
        y = np.zeros((samples,), dtype=np.float32)
        e = np.zeros((samples,), dtype=np.float32)

        for i in range(repeats):
            for t in range(1,nfram):

                idx = np.arange(self.wlen)+(t-1)*self.shift
                idx2 = np.arange(self.shift)+t*self.shift
                Bx = x[idx]
                Bd = d[idx]

                # delay line for <nslices> Fx
                self.X[:,1:] = self.X[:,:-1]
                self.X[:,0] = rfft(Bx)

                # get echo model By (overlap-save)
                Y = np.sum(self.X*self.W, axis=-1)
                By = irfft(Y)
                By = np.clip(By, -1, +1)
                y[idx2] = By[self.shift:]

                # get residual echo Be (overlap-save)
                Be = Bd - By
                Be[:self.shift] = 0                      # zero-pad lower half (overlap-save)
                e[idx2] = Be[self.shift:]
                E = rfft(Be)

                Psi_EE = np.abs(E)**2                                      # shape = (nbin,)
                Psi_WW = np.abs(self.W)**2                                      # shape = (nbin, nslices)
                Psi_XX = np.abs(self.X)**2                                      # shape = (nbin, nslices)

                # update covariance P
                tmp = np.sum(self.P*Psi_XX, axis=1) + Psi_EE + 1e-3
                mu = self.P / tmp[:,np.newaxis]                                 # shape = (nbin, nslices)
                H = np.maximum(1-Psi_XX*mu, 0.1)
                self.P = self.A*H*self.P + (1-self.A)*Psi_WW
                self.P = np.minimum(self.P, 1000)
                self.P = np.maximum(self.P, 1e-6)

                # update AEC weights W
                self.W += mu*np.conj(self.X)*E[:,np.newaxis]
                w = irfft(self.W, axis=0)
                w[self.shift:,:] = 0                    # zero-pad upper half of W (overlap-save)
                self.W = rfft(w, axis=0)
                self.W[0,:] = 0

                # shadow AEC
                Y2 = np.sum(self.X*self.W2, axis=-1)
                By2 = irfft(Y2)
                By2 = np.clip(By2, -1, +1)
                Be2 = Bd - By2
                Be2[:self.shift] = 0
                E2 = rfft(Be2)
                Bd[:self.shift] = 0
                D = rfft(Bd)

                self.Pd = self.Pd*self.alpha + (1-self.alpha)*np.sum(abs(D)**2)
                self.Pe = self.Pe*self.alpha + (1-self.alpha)*np.sum(abs(E)**2)
                self.Pe2 = self.Pe2*self.alpha + (1-self.alpha)*np.sum(abs(E2)**2)
                erle = 10*np.log10(self.Pd+1e-6) - 10*np.log10(self.Pe+1e-6)
                erle2 = 10*np.log10(self.Pd+1e-6) - 10*np.log10(self.Pe2+1e-6)
                if erle>3:
                    self.W2 = self.W
                    self.Pe2 = self.Pe
                if erle2>(erle+1):
                    self.W = self.W2
                    self.Pe = self.Pe2
                    y[idx2] = By2[self.shift:]
                    e[idx2] = Be2[self.shift:]


        e = self.constrain_magnitude(d,e)

        return e,y



    #-------------------------------------------------------------------------
    def constrain_magnitude(self, d, e):

        Fd = mstft(d)           # shape = (nfram, nbin)
        Fe = mstft(e)

        # limit magnitude of Fe to that of Fd
        Ad = np.abs(Fd)
        Ae = np.abs(Fe)
        G = np.minimum(Ad, Ae)/(Ae+1e-6)

        Fz = Fe*G
        z = mistft(Fz)

        samples = d.shape[0]
        if z.shape[0]<samples:
            pad = np.zeros((samples-z.shape[0]), dtype=np.float32)
            z = np.concatenate([z, pad])

        return z








#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    ssaec = ssaec_fast(wlen=512, tail_length=0.25)
    data = load_numpy_from_mat('../matlab/aec.mat')
    x = np.squeeze(data['x'])
    d = np.squeeze(data['d'])
    e,y = ssaec.run(x,d, repeats=2)

    data = {
            'x': x,
            'd': d,
            'e': e,
            'y': y,
           }
    save_numpy_to_mat('../matlab/aec.mat', data)




