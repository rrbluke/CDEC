# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"



import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import *
from utils.mat_helpers import *




class audio_loader(object):

    # --------------------------------------------------------------------------
    def __init__(self, path, name='audio_loader', fs=16e3):

        self.path = path
        self.fs = fs
        self.name = name

        self.file_list = glob.glob(self.path+'*.wav')
        self.numof_files = len(self.file_list)

        print('*** audio_loader "', self.name, '" found', self.numof_files, 'files')



    #-------------------------------------------------------------------------
    def cache_files(self,):

        self.x = []
        for f in self.file_list:
            s, fs = audioread(f, normalize=True)
            self.x.append(s)



    #-------------------------------------------------------------------------
    def hp_filter(self, x):

        samples = x.shape[0]
        Fx = mstft(x)
        Fx = apply_highpass_filter(Fx, self.fs, fc=100, order=4)
        x = mistft(Fx)

        if x.shape[0]<samples:
            pad = np.zeros((samples-x.shape[0]), dtype=np.float32)
            x = np.concatenate([x, pad])

        return x



    #-------------------------------------------------------------------------
    def randomize_amplitude(self, x, nband=20):

        samples = x.shape[0]
        Fx = mstft(x)

        nbin = Fx.shape[-1]
        m0 = hz_to_mel(0)
        m1 = hz_to_mel(self.fs/2)
        m = np.linspace(m0, m1, nband)
        f = mel_to_hz(m)
        G0 = np.random.uniform(-12,+12, size=(nband,))
        G0[0] = -60                       # highpass filter
        fvect = np.linspace(0,self.fs/2, nbin)
        G1 = np.interp(fvect, f, G0)
        G = np.power(10, G1/20 )
        
        Fx *= G
        x = mistft(Fx)

        if x.shape[0]<samples:
            pad = np.zeros((samples-x.shape[0]), dtype=np.float32)
            x = np.concatenate([x, pad])

        return x



    #-------------------------------------------------------------------------
    def load_random_cached_file(self, samples):

        idx = np.random.choice(len(self.file_list))
        t = np.random.randint(self.x[idx].size-samples)
        x = self.x[idx][t:t+samples]
        
        x = self.hp_filter(x)
        x *= 0.99/np.max(np.abs(x))

        return x



    #-------------------------------------------------------------------------
    def load_random_files(self, samples, offset=0):

        x = np.zeros((samples,), dtype=np.float32)
        n = offset
        while n<samples:
            f = np.random.choice(self.file_list)
            s, fs = audioread(f, normalize=True)
            length = s.shape[0]
            n1 = min(n+length, samples)
            x[n:n1] = s[0:n1-n]
            n = n1

        x = self.randomize_amplitude(x)
        x *= 0.99/np.max(np.abs(x))

        return x



