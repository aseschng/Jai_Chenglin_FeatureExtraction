#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Chenglin Xu (NTU, Singapore)

"""
Extract magnitude features, converts to TFRecords format, calculate global CMVN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from sigproc import framesig,magspec,deframesig
from scipy.signal import hamming
import scipy.io.wavfile as wav

def audioread(filename):
    (rate,sig) = wav.read(filename)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig/(max_nb_bit+1.0)
    
    return rate, sig

def normhamming(fft_len):
    if fft_len == 512:
        frame_shift = 160
    elif fft_len == 256:
        frame_shift = 128
    else:
        print("Wrong fft_len, current only support 16k/8k sampling rate wav")
        exit(1)
    win = np.sqrt(hamming(fft_len, False))
    win = win/np.sqrt(np.sum(np.power(win[0:fft_len:frame_shift],2)))
    return win

def extract(filename, FFT_LEN, FRAME_SHIFT):   

    # extract mag for mixture
    rate, sig = audioread(filename)
    frames = framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x), True)

    phase, mag_spec = magspec(frames, FFT_LEN)

    return phase, mag_spec

def reconstruct(enhan_spec, noisy_file, FFT_LEN, FRAME_SHIFT):

    (rate,sig) = wav.read(noisy_file)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig/(max_nb_bit+1.0)

    frames = framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x), True)
    phase, _ = magspec(frames, FFT_LEN)
    
    spec_comp = enhan_spec * np.exp(phase * 1j)
    enhan_frames = np.fft.irfft(spec_comp)
    enhan_sig = deframesig(enhan_frames, len(sig), FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
    enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * np.max(np.abs(sig))
    enhan_sig = enhan_sig * (max_nb_bit-1.0)
    if nb_bits == 16:
        enhan_sig = enhan_sig.astype(np.int16)
    elif nb_bits == 32:
        enhan_sig = enhan_sig.astype(np.int32)

    return enhan_sig, rate

def main():

    orig_file = 'test.wav'
    savepath = 'new_test.wav'
    FFT_LEN = 256
    FRAME_SHIFT = 128
    
    # extract magnitude features
    phase, mag_spec = extract(orig_file, FFT_LEN, FRAME_SHIFT)

    # reconstruct to wavform and save
    enhan_sig, rate = reconstruct(mag_spec, orig_file, FFT_LEN, FRAME_SHIFT)
    wav.write(savepath, rate, enhan_sig)

if __name__ == "__main__":
    main()


