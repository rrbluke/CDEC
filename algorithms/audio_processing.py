# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import os
import numpy as np
import soundfile as sf
from scipy import signal



#----------------------------------------------------------------
# read multichannel audio data
# output x.shape = (samples, nmic)
def audioread(filename, normalize=False, subtract_mean=True):

    x, fs = sf.read(filename)

    if subtract_mean==True:
        x -= np.mean(x, axis=0, keepdims=True)

    if normalize==True:
        x = x*0.99/(np.max(np.abs(x)) + 1e-3)

    return (x, fs)


#----------------------------------------------------------------
# write multichannel audio data
# input x.shape = (samples, nmic)
def audiowrite(x, filename, fs=16000, normalize=False):

    #x.shape = (samples, channels)
    if normalize==True:
        x = x*0.99/(np.max(np.abs(x)) + 1e-3)

    sf.write(filename, x, fs)



#----------------------------------------------------------------
# wrapper for python real fft
def rfft(Bx, n=None, axis=-1):

    Fx = np.fft.rfft(Bx, n=n, axis=axis).astype(np.complex64)

    return Fx


#----------------------------------------------------------------
# wrapper for python real ifft
def irfft(Fx, n=None, axis=-1):

    Bx = np.fft.irfft(Fx, n=None, axis=axis).astype(np.float32)

    if n is not None:
        samples = Bx.shape[axis]
        Bx = np.moveaxis(Bx, axis, 0)
        if n<samples:
            Bx = Bx[:n,...]
        else:
            shape = (n,) + Bx.shape[1:]
            By = np.zeros(shape=shape, dtype=Bx.dtype)
            By[:samples,...] = Bx
        Bx = np.moveaxis(By, 0, axis)

    return Bx
    


#----------------------------------------------------------------
def create_hanning_window(wlen):

    t = np.arange(wlen)
    window = 0.5*(1-np.cos(2*np.pi*(t+1)/(wlen+1)))
    window = np.sqrt(window)

    return window



#----------------------------------------------------------------
# perform the STFT on audio data x
# x.shape = (samples,)
# output Fx.shape = (nfram, nbin)
def mstft(x, wlen=1024):

    assert (x.ndim == 1), 'x must have 1 dimension'

    samples = x.shape[0]
    shift = wlen//2
    nbin = wlen//2+1
    nfram = samples//shift

    x = np.asarray(x, dtype=np.float32)
    pad = np.zeros((wlen-shift,), dtype=np.float32)
    x = np.concatenate([pad,x], axis=0)
    window = create_hanning_window(wlen)

    Fx = np.zeros((nfram, nbin), dtype=np.complex64)
    for t in range(nfram):
        idx = np.arange(wlen) + t*shift
        Bx = x[idx] * window
        Fx[t,:] = np.fft.rfft(Bx)

    return Fx



#----------------------------------------------------------------
# perform the inverse STFT on audio data Fx
# Fx.shape = (nfram, nbin)
# output x.shape = (samples,)
def mistft(Fx, wlen=1024):

    assert (Fx.ndim == 2), 'Fx must have 2 dimensions'

    shift = wlen//2
    nbin = Fx.shape[1]
    nfram = Fx.shape[0]
    samples = nfram*shift+wlen-shift

    Fx = np.asarray(Fx, dtype=np.complex64)
    window = create_hanning_window(wlen)

    if Fx.ndim == 2:
        x = np.zeros((samples,), dtype=np.float32)
        for t in range(nfram):
            Bx = np.real(np.fft.irfft(Fx[t,:]))
            idx = np.arange(wlen) + t*shift
            x[idx] += Bx * window

    return x[(wlen-shift):]



#------------------------------------------------------------------------------
# get amplitude response of a highpass filter with <order> 
# fs = samplerate
# fc = corner frequency
# response at H(fc) = 1/sqrt(2)
# nbin = number of frequency bins
def get_highpass_filter(nbin=513, fs=16e3, fc=100, order=2):

    f = np.arange(nbin)*fs/(2*(nbin-1))
    k = 1/(np.sqrt(2)-1)
    tmp = k*np.power(f/fc, order)
    H = tmp/(1+tmp)

    return H



#------------------------------------------------------------------------------
# get amplitude response of a lowpass filter with <order> 
# fs = samplerate
# fc = corner frequency
# response at H(fc) = 1/sqrt(2)
# nbin = number of frequency bins
def get_lowpass_filter(nbin=513, fs=16e3, fc=100, order=2):

    f = np.arange(nbin)*fs/(2*(nbin-1))
    k = 1/(np.sqrt(2)-1)
    tmp = k*np.power(fc/np.maximum(f, 1e-6), order)
    H = tmp/(1+tmp)

    return H



#------------------------------------------------------------------------------
def apply_highpass_filter(Fx, fs=16e3, fc=100, order=2):

    nbin = Fx.shape[-1]

    H = get_highpass_filter(nbin=nbin, fs=fs, fc=fc, order=order)
    Fz = np.einsum('...k,k->...k', Fx, H)

    return Fz



#------------------------------------------------------------------------------
def apply_lowpass_filter(Fx, fs=16e3, fc=100, order=2):

    nbin = Fx.shape[-1]

    H = get_lowpass_filter(nbin=nbin, fs=fs, fc=fc, order=order)
    Fz = np.einsum('...k,k->...k', Fx, H)

    return Fz



#------------------------------------------------------------------------------
def hz_to_mel(hz):

    return 2595*np.log10(1+hz/700)



#------------------------------------------------------------------------------
def mel_to_hz(mel):

    return 700*(10**(mel/2595)-1)



#------------------------------------------------------------------------------
def create_mel_filterbank(nbin=513, fs=16e3, nband=40):

    low_freq_mel = 100
    high_freq_mel = hz_to_mel(fs/2-100)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nband+2, dtype=np.float32)        # equally spaced mel scale with <nband> kernels
    hz_points = mel_to_hz(mel_points)

    bin_index = np.asarray(np.floor(2*nbin*hz_points/fs), dtype=np.int32)
    filterbank = np.zeros((nband, nbin), dtype=np.float32)

    for m in range(1, nband+1):

        # create <nband> triangular kernels
        f_m_left = bin_index[m-1]
        f_m_center = bin_index[m]
        f_m_right = bin_index[m+1]

        for k in range(f_m_left, f_m_center):
            filterbank[m-1, k] = (k-f_m_left) / (f_m_center-f_m_left)

        for k in range(f_m_center, f_m_right):
            filterbank[m-1, k] = (f_m_right-k) / (f_m_right-f_m_center)

    return filterbank                  # shape = (nband, nbin)



#------------------------------------------------------------------------------
def mkdir(path):

    if not os.path.exists(os.path.dirname(path)): 
        os.makedirs(os.path.dirname(path))


