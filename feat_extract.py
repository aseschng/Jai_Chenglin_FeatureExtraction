  #!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Chng Eng Siong
# extending Chenglin's code to extract features and generate signal
# we will use python_speech_feature directly
# as well as various matplotlib to show the signal of interest

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import  python_speech_features as psf
import  scipy.io.wavfile as wav
import  matplotlib.pyplot as plt
import  numpy as np
from    scipy.signal import hamming


# This function uses scipy's wavfile routine to read.
# unfortunately scipy's function is primitive,
# we want to return a normalized value +- 1.0 in amplitude for processing
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
    frames = psf.sigproc.framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
#    frames = framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
#    phase, mag_spec = magspec(frames, FFT_LEN)
    complex_spec = np.fft.rfft(frames, FFT_LEN)
    phase        = np.angle(complex_spec)
    mag_spec     = np.absolute(complex_spec)

    return phase, mag_spec, rate, sig


def reconstruct(enhan_spec, phase, FFT_LEN, FRAME_SHIFT):

  # following is a sanity check, realising why ONLY strong spectral values
  # phase are important, and the rest we can set to zero. :)
  threshold_val = 0.001*enhan_spec.max()
  threshold     = threshold_val*np.ones(enhan_spec.shape)
  my_mask       = ((enhan_spec - threshold) > 0)
  my_phase      = phase * my_mask
  print('max enhan spec = ', enhan_spec.max(), 'and threshold =', threshold_val)
  nr,nc = my_mask.shape
  numElements = nr*nc
  print('number of spectra bin > threshold = ', my_mask.sum(),' as percentage = ',my_mask.sum()/numElements)

  fig, ax = plt.subplots(2, 1)
  ax[0].plot(my_mask)
  ax[1].plot(my_phase)
  plt.show()

  spec_comp = enhan_spec * np.exp(my_phase * 1j)
  nb_bits = 16
  enhan_frames = np.fft.irfft(spec_comp)
  enhan_sig = psf.sigproc.deframesig(enhan_frames, 0, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
  enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * 0.8
  # above is simply to get the amplitude to normalize to some reasonable value
  max_nb_bit = float(2 ** (nb_bits - 1))
  enhan_sig = enhan_sig * (max_nb_bit - 1.0)
  if nb_bits == 16:
      enhan_sig = enhan_sig.astype(np.int16)
  elif nb_bits == 32:
      enhan_sig = enhan_sig.astype(np.int32)

  return enhan_sig


def main():

    orig_file = 'test.wav'
    sample_rate  = 8000
    FFT_LEN      = 256
    FRAME_SHIFT  = 128

    # extract magnitude features
    phase, mag_spec, rate, sig = extract(orig_file, FFT_LEN, FRAME_SHIFT)
    nb_bits = 16
    print('Signa datatype = ', sig.dtype)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32

    fig,ax = plt.subplots(1,1)
    t = np.arange(0, len(sig), dtype=np.float)/ sample_rate
    ax.plot(t,sig)
    plt.show()

    mag_spec_np = np.copy(mag_spec.transpose())
    (freq_size, frame_num) = mag_spec_np.shape
    t2frame = np.arange(0, frame_num, dtype=np.float) * freq_size/ sample_rate

    fig,ax = plt.subplots(2,1)
    ax[0].plot(t2frame,np.median(mag_spec_np,axis=0))
    ax[0].set_xlim(0, t2frame[-1])

    # refernce to plot spectryum
    # https://lo.calho.st/projects/generating-a-spectrogram-using-numpy/
    # reference for matlab plotting axes, see:
    #

    Y = (20 * np.log10(mag_spec_np)).clip(-90)
    f = np.arange(0, freq_size, dtype=np.float) * sample_rate / (2*freq_size)

    ax[1].pcolormesh(t2frame, f, Y, vmin=-90, vmax=0)
    #    ax[1].set_yscale('symlog', linthreshy=100, linscaley=0.25)
    ax[1].set_xlim(0, t2frame[-1])
    ax[1].set_ylim(0, f[-1])

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")

    # cbar = plt.colorbar()
    # cbar.set_label("Intensity (dB)")
    plt.show()

    # reconstruct to wavform and save
    enhan_sig = reconstruct(mag_spec, phase,  FFT_LEN, FRAME_SHIFT)
    savepath = 'new_test.wav'
    wav.write(savepath, rate, enhan_sig)

if __name__ == "__main__":
    main()


