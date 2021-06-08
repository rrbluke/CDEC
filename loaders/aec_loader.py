# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"



import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath('../'))

from algorithms.audio_processing import *
from utils.mat_helpers import *




class aec_loader(object):

    #--------------------------------------------------------------------------
    def __init__(self, name='aec_loader', dataset_dir='../../Interspeech_AEC_Challenge_2021/AEC-Challenge/datasets/', fs=16e3):

        self.fs = fs
        self.name = name
        self.dataset_dir = dataset_dir

        # load train_real files
        train_real_path = self.dataset_dir+'real/'
        #x_train_real = glob.glob(train_real_path+'*_farend_singletalk_with_movement_lpb.wav')
        #d_train_real = glob.glob(train_real_path+'*_farend_singletalk_with_movement_mic.wav')
        x_train_real = glob.glob(train_real_path+'*_farend_singletalk_*lpb.wav')
        d_train_real = glob.glob(train_real_path+'*_farend_singletalk_*mic.wav')
        self.d_train_real = []
        for x in x_train_real:
            d = x.replace('lpb', 'mic')
            if d in d_train_real:
                self.d_train_real.append(d)

        # load train_simu files
        train_simu_path = self.dataset_dir+'synthetic/'
        x_train_simu = glob.glob(train_simu_path+'farend_speech/farend_speech_fileid_*.wav')
        d_train_simu = glob.glob(train_simu_path+'echo_signal/echo_fileid_*.wav')
        self.x_train_simu = []
        self.d_train_simu = []
        for x in x_train_simu:
            d = x.replace('farend_speech/farend_speech', 'echo_signal/echo')
            if d in d_train_simu:
                self.x_train_simu.append(x)
                self.d_train_simu.append(d)

        # load train_hard files
        train_hard_path = self.dataset_dir+'train_hard/'
        self.d_train_hard = glob.glob(train_hard_path+'*/*mic.wav')

        # load test files
        self.test_path = self.dataset_dir+'test_set_interspeech2021/'
        self.d_test = glob.glob(self.test_path+'*/*mic.wav')

        # load test_blind files
        self.test_blind_path = self.dataset_dir+'blind_test_set_interspeech2021/'
        self.d_test_blind = glob.glob(self.test_blind_path+'*/*mic.wav')


        print('*** audio loader "', self.name, '" found', len(self.d_train_real), 'train_real files')
        print('*** audio loader "', self.name, '" found', len(self.d_train_simu), 'train_simu files')
        print('*** audio loader "', self.name, '" found', len(self.d_train_hard), 'train_hard files')
        print('*** audio loader "', self.name, '" found', len(self.d_test), 'test files')
        print('*** audio loader "', self.name, '" found', len(self.d_test_blind), 'test_blind files')

        quit()



    #-------------------------------------------------------------------------
    def hp_filter(self, x):

        Fx = mstft(x)
        Fx = apply_highpass_filter(Fx, self.fs, fc=100, order=4)
        x = mistft(Fx)

        return x



    #-------------------------------------------------------------------------
    def lp_filter(self, x):

        Fx = mstft(x)
        Fx = apply_lowpass_filter(Fx, self.fs, fc=7500, order=8)
        x = mistft(Fx)

        return x



    #-------------------------------------------------------------------------
    def load_train(self, mode, idx=None):

        if mode == 'real':
            return self.load_train_real(idx)

        elif mode == 'simu':
            return self.load_train_simu(idx)

        elif mode == 'hard':
            return self.load_train_hard(idx)



    #-------------------------------------------------------------------------
    def load_train_real(self, idx=None):

        if idx is not None:
            idx = np.mod(idx, len(self.d_train_real))
        else:
            idx = np.random.choice(len(self.d_train_real))

        valid = False
        while valid is False:

            name = self.d_train_real[idx].replace('mic.wav', '')
            x, fs = audioread(name+'lpb.wav')
            d, fs = audioread(name+'mic.wav')

            x = self.hp_filter(x)
            d = self.hp_filter(d)
            d = self.lp_filter(d)

            samples = min(len(x), len(d))
            x = x[:samples]
            d = d[:samples]

            Pd = 10*np.log10(np.mean(d**2))
            if Pd > -40:
                valid = True
            else:
                idx = np.random.choice(len(self.d_train_real))

        Pd = 10*np.log10(np.mean(d**2))
        G = np.power(10, (-26-Pd)/20)
        x *= G
        d *= G

        return x,d



    #-------------------------------------------------------------------------
    def load_train_simu(self, idx=None):

        if idx is not None:
            idx = np.mod(idx, len(self.x_train_simu))
        else:
            idx = np.random.choice(len(self.x_train_simu))

        valid = False
        while valid is False:

            x, fs = audioread(self.x_train_simu[idx])
            d, fs = audioread(self.d_train_simu[idx])

            x = self.hp_filter(x)
            d = self.hp_filter(d)
            d = self.lp_filter(d)

            samples = min(len(x), len(d))
            x = x[:samples]
            d = d[:samples]

            Pd = 10*np.log10(np.mean(d**2))
            if Pd > -40:
                valid = True
            else:
                idx = np.random.choice(len(self.x_train_simu))

        Pd = 10*np.log10(np.mean(d**2))
        G = np.power(10, (-26-Pd)/20)
        x *= G
        d *= G

        return x,d



    #-------------------------------------------------------------------------
    def load_train_hard(self, idx=None):

        if idx is not None:
            idx = np.mod(idx, len(self.d_train_hard))
        else:
            idx = np.random.choice(len(self.d_train_hard))

        name = self.d_train_hard[idx].replace('mic.wav', '')
        x, fs = audioread(name+'lpb.wav')
        d, fs = audioread(name+'mic.wav')

        x = self.hp_filter(x)
        d = self.hp_filter(d)
        d = self.lp_filter(d)

        samples = min(len(x), len(d))
        x = x[:samples]
        d = d[:samples]

        Pd = 10*np.log10(np.mean(d**2))
        G = np.power(10, (-26-Pd)/20)
        x *= G
        d *= G

        return x,d



    #-------------------------------------------------------------------------
    def load_test(self, idx):

        name = self.d_test[idx].replace('mic.wav', '')
        x, fs = audioread(name+'lpb.wav')
        d, fs = audioread(name+'mic.wav')

        x = self.hp_filter(x)
        d = self.hp_filter(d)
        d = self.lp_filter(d)

        samples = min(len(x), len(d))
        x = x[:samples]
        d = d[:samples]

        G = 0.99/np.max(np.abs(d))
        x *= G
        d *= G

        return x,d



    #-------------------------------------------------------------------------
    def load_test_blind(self, idx):

        name = self.d_test_blind[idx].replace('mic.wav', '')
        x, fs = audioread(name+'lpb.wav')
        d, fs = audioread(name+'mic.wav')

        x = self.hp_filter(x)
        d = self.hp_filter(d)
        d = self.lp_filter(d)

        samples = min(len(x), len(d))
        x = x[:samples]
        d = d[:samples]

        G = 0.99/np.max(np.abs(d))
        x *= G
        d *= G

        return x,d



    #-------------------------------------------------------------------------
    def find_idx_from_testset(self, name):

        for idx,f in enumerate(self.d_test):
            if name in f:
                return idx

        return None



    #-------------------------------------------------------------------------
    def find_idx_from_blind_testset(self, name):

        for idx,f in enumerate(self.d_test_blind):
            if name in f:
                return idx

        return None



    #-------------------------------------------------------------------------
    def write_enhanced(self, x, idx, subfolder):

        G = 0.99/np.max(np.abs(x))
        x *= np.minimum(G, 1)

        name = self.d_test[idx].replace('mic.wav', 'enh.wav')
        name = name.replace(self.test_path, '')
        name = self.dataset_dir+'submission/'+subfolder+'/'+name
        mkdir(name)
        audiowrite(x, name)



    #-------------------------------------------------------------------------
    def write_enhanced_blind(self, x, idx, subfolder):

        G = 0.99/np.max(np.abs(x))
        x *= np.minimum(G, 1)

        name = self.d_test_blind[idx].replace('mic.wav', 'enh.wav')
        name = name.replace(self.test_blind_path, '')
        name = self.dataset_dir+'submission_blind/'+subfolder+'/'+name
        mkdir(name)
        audiowrite(x, name)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    loader = aec_loader()
    idx = loader.find_idx_from_testset('0cz6i0nuEU0WwRnhrNVxBw')
    print(idx)
    x,d = loader.load_test(idx)

    data = {
            'x': x,
            'd': d,
           }
    save_numpy_to_mat('../matlab/fgen_check.mat', data)

