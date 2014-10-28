# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:02:36 2014

@author: yves-laurent
"""

from matplotlib.pyplot import plot, show, xlabel, ylabel, subplot, figure
from scipy import fft, arange, ifft
from scipy.io import wavfile
from scipy.signal import convolve
import scipy.io as sio
import numpy as np


def al_play(snd, fs):
    import scikits.audiolab as al
    import numpy as np
    from scipy.signal import resample
    if len(snd.shape) == 2:
        if snd.shape[1] == 2 or snd.shape[1] == 1:
            snd = snd.T
        elif snd.shape[0] != 2 and snd.shape[0] != 1:
            print "sound must either a vector or a rank 2 matrix"
            return
        N = snd.shape[1]
    else:
        N = snd.shape[0]
        snd = snd.reshape(-1, N)
    rsmple = resample(snd, N * 44100.0 / (1.0 * fs), axis=1)
    al.play(rsmple, fs=44100)


def plotSpectrum(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    :param y: the signal
    :param Fs: the sampling frequency
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs 
    frq = k/T # Two sides frequency range
    frq = np.squeeze(frq)

    frq = frq[range(int(n/2))] # One side frequency range

    Y = fft(y)/n # FFT computing and normalization
    Y = Y[range(int(n/2))]
    # Plot the signal in wall-clock time
    subplot(2,1,1)
    Ts = 1.0/Fs; # sampling interval
    t = arange(0, 1.0*len(y)/Fs, Ts)
    plot(t, y)
    xlabel('Time')
    ylabel('Amplitude')
    # Plot the spectrum 
    subplot(2,1,2)
    plot( frq, np.log(abs(Y)),'r') # Plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('log|Y(freq)|')
    show()      
    

def saveWav(y, name, Fs=44100):
    """
    Save a signal y as a wave file
    :param y: the signal
    :param name: the name of the file
    :param rate: the sampling frequency
    """
    wavfile.write(name, Fs, y)


def saveMatasWav():
    mc = sio.loadmat('hum_remove.mat')
    snoisy = mc['s_noisy']
    saveWav(snoisy, "file.wav", Fs=9000)



def f_cheat(y, fs):
    """
    y : the noisy signal
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/fs
    frq = k/T # Two sides frequency range
    frq = np.squeeze(frq)
    frq = frq[range(int(n/2))] # One side frequency range

    half = int(n/2)
    Y = fft(y) # FFT computing and normalization
    l,u = 6550, 6750
    Y[l:u] = 0.0
    Y[n-u : n-l] = 0.0
    y_clean = np.real(ifft(Y))
    return y_clean


def lowpass(y):
    """
    y = 1/3[x0 + x-1 + x-2]
    """
    return convolve(y, np.ones(3)/3.0, mode='same')


"""
Below are a couple of code snippets. When needed, make sure that the audio file
is on your Python path (typically you might want to put it in the same directory as this file).
"""

"""
Snippet 1: Read a .wav file. Uncomment the below to use.
"""
#file_name = 'file.wav'
#Fs, y = wavfile.read(file_name)

"""
Snippet 2: Plot a single-sided spectrum. Uncomment the below to use.
# """
# saveMatasWav()
# file_name = 'file.wav'
# Fs, y = wavfile.read(file_name)
# plotSpectrum(y, Fs)
mc = sio.loadmat('hum_remove.mat')
clean_signal = mc['s'][:,0]
noisy_signal = mc['s_noisy'][:,0]
fs = mc['fs']
#al_play(clean_signal,fs)
y_clean   = f_cheat(noisy_signal, fs)
y_lowpass = lowpass(noisy_signal)
#plotSpectrum(y_clean, fs)
#plotSpectrum(y_lowpass, fs)

#plotSpectrum(y_clean, fs)

al_play(y_lowpass, fs)
#al_play(clean_signal, fs)

#ss= np.abs(np.log(fft(clean_signal))) - np.abs(np.log(fft(y_clean)))
#plot(ss[::10])
#show()



"""
Snippet 3: Save a .wav file. Uncomment the below to use, run and check the current directory.
"""
#file_name = 'file.wav'
#Fs, y = wavfile.read(file_name)
#y_noisy = y + np.random.randn(len(y))
#saveWav(y, 'noisy_' + file_name, Fs=Fs)

"""
Snippet 4: Recover a signal from its DFT. Uncomment the below to use.
"""
#file_name = 'file.wav'
#Fs, y = wavfile.read(file_name)
#F = fft(y)
#y_recov = ifft(F)




