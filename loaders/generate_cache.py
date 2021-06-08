# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))

from loaders.aec_loader import aec_loader
from loaders.audio_loader import audio_loader
from utils.mat_helpers import *
from algorithms.audio_processing import *
from algorithms.ssaec_fast import *




class generate_cache(object):

    # --------------------------------------------------------------------------
    def __init__(self,):

        self.aec_loader = aec_loader(name='aec_loader')
        self.dt_loader = audio_loader(name='doubletalk', path='../../wsj0/si_tr_*/*/')
        self.noise_loader = audio_loader(name='noise', path='../../youtube_noise/*/')
        self.noise_loader.cache_files()
        self.ssaec = ssaec_fast(wlen=512, tail_length=0.250)

        self.timestamp = os.path.getmtime(sys.argv[0])                               # timestamp of this script

        self.fs = self.aec_loader.fs
        self.samples = int(self.fs*15)
        self.silence = int(self.fs*5)

        self.dataset_dir = self.aec_loader.dataset_dir
        self.scenarios = ['nearend', 'farend', 'doubletalk']
        self.modes = ['real','simu','hard']
        self.train_set_length = 10000
        self.test_set_length = len(self.aec_loader.d_test)
        self.blind_test_set_length = len(self.aec_loader.d_test_blind)



    #-------------------------------------------------------------------------
    def pad_files(self, x, d):

        samples = len(x)
        x0 = np.zeros((self.samples,), dtype=np.float32)
        d0 = np.zeros((self.samples,), dtype=np.float32)

        if samples>self.samples:
            x0 = x[-self.samples:]
            d0 = d[-self.samples:]
        else:
            n = 0
            while n<self.samples:
                n1 = min(n+samples, self.samples)
                x0[n:n1] = x[0:n1-n]
                d0[n:n1] = d[0:n1-n]
                n = n1

        return x0,d0



    #-------------------------------------------------------------------------
    def compensate_delay(self, x, d):

        Fx = rfft(x)
        Fd = rfft(d)

        Phi = Fd*np.conj(Fx)
        Phi /= np.abs(Phi) + 1e-3
        Phi[0] = 0
        tmp = irfft(Phi)
        tau = np.argmax(np.abs(tmp))
        x = np.roll(x, tau)
        #print(tau/self.fs)

        return x



    #-------------------------------------------------------------------------
    def generate_train(self, mode, scenario, idx):

        x,d = self.aec_loader.load_train(mode, idx)
        x,d = self.pad_files(x,d)
        samples3 = self.samples//3

        if mode == 'simu':

            # create artificial delay changes by shifting the last third randomly
            lag = int(np.random.uniform(-0.020, 0)*self.fs)
            x[2*samples3:self.samples] = np.roll(x[2*samples3:self.samples], lag)


        if mode in ['simu','real']:

            # randomly attenuate microphone signal in the middle third
            G = np.ones((self.samples,), dtype=np.float32)
            G[samples3:2*samples3] = np.power(10, np.random.uniform(0,-40)/20 )
            d = d*G

            # randomly add clipping
            G = np.power(10, np.random.uniform(0,+12)/20 )
            d = np.clip(d*G, -1, +1)/G


        # load doubletalk
        s = self.dt_loader.load_random_files(samples=self.samples, offset=self.silence)

        # load noise
        n = self.noise_loader.load_random_cached_file(samples=self.samples)

        Pd = 10*np.log10(np.mean(d**2))
        Ps = 10*np.log10(np.mean(s**2))
        Pn = 10*np.log10(np.mean(n**2))
        sir = np.random.uniform(0,+6)               # add doubletalk at 0...+6dB SIR
        snr = np.random.uniform(12,32)              # add noise at 12...32dB SNR
        s *= np.power(10, (Pd-Ps+sir)/20)
        n *= np.power(10, (Pd-Pn-snr)/20)

        # do not add noise to real or hard files
        if mode in ['real','hard']:
            n = np.zeros_like(d)

        # simulate scenario
        if scenario=='nearend':
            x = np.zeros_like(d)
            d = s+n

        if scenario=='farend':
            s = np.zeros_like(d)
            d = d+n
            
        if scenario=='doubletalk':
            d = d+s+n

        # compensate bulk delay
        x = self.compensate_delay(x,d)

        # perform EC with random starting position
        start = np.random.choice(self.samples)
        x0 = np.roll(x, start)
        d0 = np.roll(d, start)
        e,y = self.ssaec.run(x0, d0, repeats=2)
        e = np.roll(e, -start)
        y = np.roll(y, -start)

        return x,y,d,e,s



    #-------------------------------------------------------------------------
    def generate_test(self, idx):

        x,d = self.aec_loader.load_test(idx)
        x = self.compensate_delay(x,d)
        e,y = self.ssaec.run(x, d, repeats=2)

        return x,y,d,e



    #-------------------------------------------------------------------------
    def generate_blind_test(self, idx):

        x,d = self.aec_loader.load_test_blind(idx)
        x = self.compensate_delay(x,d)
        e,y = self.ssaec.run(x, d, repeats=2)

        return x,y,d,e



    #-------------------------------------------------------------------------
    def cache_train_set(self,):

        for mode in np.random.permutation(self.modes):
            for scenario in np.random.permutation(self.scenarios):
                for idx in np.random.permutation(range(self.train_set_length)):

                    name = self.dataset_dir+'train_cache/'+mode+'/'+scenario+'/'+'{:04d}'.format(idx)+'.mat'
                    if os.path.isfile(name):
                        if self.timestamp < os.path.getmtime(name):
                            continue

                    x,y,d,e,s = self.generate_train(mode, scenario, idx)
                    data = {
                        'x': x,
                        'y': y,
                        'd': d,
                        'e': e,
                        's': s,
                        'fs': self.fs,
                        'mode': mode,
                        'scenario': scenario,
                        }
                    mkdir(name)
                    save_numpy_to_mat(name, data)
                    print('writing train file:', idx, '/', self.train_set_length)



    #-------------------------------------------------------------------------
    def cache_test_set(self,):

        for idx in np.random.permutation(range(self.test_set_length)):

            name = self.dataset_dir+'test_cache/'+'{:04d}'.format(idx)+'.mat'
            if os.path.isfile(name):
                if self.timestamp < os.path.getmtime(name):
                    continue

            x,y,d,e = self.generate_test(idx)
            data = {
                'x': x,
                'y': y,
                'd': d,
                'e': e,
                'fs': self.fs,
                'idx': idx,
                }
            mkdir(name)
            save_numpy_to_mat(name, data)
            print('writing test file:', idx, '/', self.test_set_length)



    #-------------------------------------------------------------------------
    def cache_blind_test_set(self,):

        for idx in np.random.permutation(range(self.blind_test_set_length)):

            name = self.dataset_dir+'blind_test_cache/'+'{:04d}'.format(idx)+'.mat'
            if os.path.isfile(name):
                if self.timestamp < os.path.getmtime(name):
                    continue

            x,y,d,e = self.generate_blind_test(idx)
            data = {
                'x': x,
                'y': y,
                'd': d,
                'e': e,
                'fs': self.fs,
                'idx': idx,
                }
            mkdir(name)
            save_numpy_to_mat(name, data)
            print('writing blind test file:', idx, '/', self.blind_test_set_length)





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    gc = generate_cache()
    gc.cache_train_set()
    gc.cache_test_set()
    gc.cache_blind_test_set()


