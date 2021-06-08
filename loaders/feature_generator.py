# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))

from loaders.aec_loader import aec_loader
from utils.mat_helpers import *
from algorithms.audio_processing import *
from algorithms.ssaec_fast import *




class feature_generator(object):

    # --------------------------------------------------------------------------
    def __init__(self,):

        self.aec_loader = aec_loader(name='aec')
        self.ssaec = ssaec_fast(wlen=512, tail_length=0.250)

        self.fs = self.aec_loader.fs
        self.samples = int(self.fs*15)
        self.silence = int(self.fs*5)

        self.dataset_dir = self.aec_loader.dataset_dir
        self.scenarios = ['nearend', 'farend', 'doubletalk']
        self.modes = ['real','simu','hard']
        self.train_set_length = 5000
        self.test_set_length = len(self.aec_loader.d_test)
        self.blind_test_set_length = len(self.aec_loader.d_test_blind)

        self.nband = 25
        self.Q_long = create_mel_filterbank(nbin=513, fs=16e3, nband=self.nband)
        self.Q_short = create_mel_filterbank(nbin=129, fs=16e3, nband=self.nband)



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
    #scenarios = ['nearend', 'farend', 'doubletalk']
    #modes = ['real','simu','hard']
    def load_train(self, nbatch=1, mode=None, scenario=None, idx=None, p_modes=[0.3, 0.3, 0.4], p_scenarios=[0.1, 0.1, 0.8]):

        mode0 = mode
        scenario0 = scenario
        idx0 = idx

        x = np.zeros((nbatch, self.samples-self.silence), dtype=np.float32)
        y = np.zeros((nbatch, self.samples-self.silence), dtype=np.float32)
        d = np.zeros((nbatch, self.samples-self.silence), dtype=np.float32)
        e = np.zeros((nbatch, self.samples-self.silence), dtype=np.float32)
        s = np.zeros((nbatch, self.samples-self.silence), dtype=np.float32)

        for b in range(nbatch):

            if mode0==None:
                mode = np.random.choice(self.modes, p=p_modes)
            else:
                mode = mode0

            if scenario0==None:
                scenario = np.random.choice(self.scenarios, p=p_scenarios)
            else:
                scenario = scenario0

            if idx0==None:
                idx = np.random.choice(self.train_set_length)
            else:
                idx = idx0
            name = self.dataset_dir+'train_cache/'+mode+'/'+scenario+'/'+'{:04d}'.format(idx)+'.mat'
            data = load_numpy_from_mat(name)

            x[b,:] = data['x'][0,self.silence:]
            y[b,:] = data['y'][0,self.silence:]
            d[b,:] = data['d'][0,self.silence:]
            e[b,:] = data['e'][0,self.silence:]
            s[b,:] = data['s'][0,self.silence:]

        return x,y,d,e,s



    #-------------------------------------------------------------------------
    def load_test(self, idx, nbatch=1):

        name = self.dataset_dir+'test_cache/'+'{:04d}'.format(idx)+'.mat'
        data = load_numpy_from_mat(name)

        x = data['x'][0,:]         # shape = (1, samples)
        y = data['y'][0,:]         # shape = (1, samples)
        d = data['d'][0,:]         # shape = (1, samples)
        e = data['e'][0,:]         # shape = (1, samples)

        x = np.stack([x]*nbatch, axis=0)
        y = np.stack([y]*nbatch, axis=0)
        d = np.stack([d]*nbatch, axis=0)
        e = np.stack([e]*nbatch, axis=0)
        s = np.zeros_like(x)

        return x,y,d,e,s



    #-------------------------------------------------------------------------
    def load_test_blind(self, idx, nbatch=1):

        name = self.dataset_dir+'blind_test_cache/'+'{:04d}'.format(idx)+'.mat'
        data = load_numpy_from_mat(name)

        x = data['x'][0,:]         # shape = (1, samples)
        y = data['y'][0,:]         # shape = (1, samples)
        d = data['d'][0,:]         # shape = (1, samples)
        e = data['e'][0,:]         # shape = (1, samples)

        x = np.stack([x]*nbatch, axis=0)
        y = np.stack([y]*nbatch, axis=0)
        d = np.stack([d]*nbatch, axis=0)
        e = np.stack([e]*nbatch, axis=0)
        s = np.zeros_like(x)

        return x,y,d,e,s



    #-------------------------------------------------------------------------
    def write_enhanced(self, x, idx, experiment_name):

        # batchsize = 1
        self.aec_loader.write_enhanced(x[0,:], idx, experiment_name)



    #-------------------------------------------------------------------------
    def write_enhanced_blind(self, x, idx, experiment_name):

        # batchsize = 1
        self.aec_loader.write_enhanced_blind(x[0,:], idx, experiment_name)



    #-------------------------------------------------------------------------
    def load_valid(self, nbatch):

        idx = self.aec_loader.find_idx_from_testset('8WdP6ehkpkiy5AHUg0DnNg_doubletalk_with_movement')
        name = self.dataset_dir+'test_cache/'+'{:04d}'.format(idx)+'.mat'
        data = load_numpy_from_mat(name)

        x = data['x'][0,:]         # shape = (samples,)
        y = data['y'][0,:]         # shape = (samples,)
        d = data['d'][0,:]         # shape = (samples,)
        e = data['e'][0,:]         # shape = (samples,)

        x = np.stack([x]*nbatch, axis=0)
        y = np.stack([y]*nbatch, axis=0)
        d = np.stack([d]*nbatch, axis=0)
        e = np.stack([e]*nbatch, axis=0)
        s = np.zeros_like(x)

        return x,y,d,e,s



    #-------------------------------------------------------------------------
    def write_aec_only(self,):

        for idx in range(self.test_set_length):
            name = self.dataset_dir+'test_cache/'+'{:04d}'.format(idx)+'.mat'
            data = load_numpy_from_mat(name)
            e = data['e'][0,:]                  # shape = (samples,)
            self.aec_loader.write_enhanced(e, idx, 'aec_only')
            print('writing file:', idx, '/', self.test_set_length)



    #-------------------------------------------------------------------------
    def write_aec_only_blind(self,):

        for idx in range(self.blind_test_set_length):
            name = self.dataset_dir+'blind_test_cache/'+'{:04d}'.format(idx)+'.mat'
            data = load_numpy_from_mat(name)
            e = data['e'][0,:]                  # shape = (samples,)
            self.aec_loader.write_enhanced_blind(e, idx, 'aec_only')
            print('writing file:', idx, '/', self.blind_test_set_length)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    fgen = feature_generator()

    #fgen.write_aec_only_blind()
    #quit()
    
    Q1 = create_mel_filterbank(nbin=513, fs=16e3, nband=25)
    Q2 = create_mel_filterbank(nbin=129, fs=16e3, nband=25)

    data = {
            'Q1': Q1,
            'Q2': Q2,
           }
    save_numpy_to_mat('../matlab/test.mat', data)
    quit()


    #t0 = time.time()
    #x,y,d,e,s = fgen.load_train(3)
    #t1 = time.time()
    #print(t1-t0)
    #quit()

    idx = fgen.aec_loader.find_idx_from_testset('n2MtzNsbnkudT1iT1oZXNQ')
    x,y,d,e,s = fgen.load_test(idx)

    data = {
            'y': y,
            'x': x,
            'd': d,
            'e': e,
            's': s,
           }
    save_numpy_to_mat('../matlab/fgen_check.mat', data)




