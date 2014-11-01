import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy import interpolate
import os.path as osp
from scipy.signal import butter, filtfilt, find_peaks_cwt


ddir = "/Users/ankushgupta/cdt_courses/signal_proc/final_project"


def low_pass(x,  fc, fs):
    """
    Removes high-frequency content from an array X.
    fs : sampling frequency
    """
    lowcut  = fc   # lower-most freq (Hz)
    highcut = 3.0   # higher-most freq (Hz)

    nyq = 0.5 * fs  # nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, high, btype='low')
    
    x_filt = filtfilt(b, a, x, axis=0)
    x_filt = np.squeeze(x_filt)
    return  x_filt

def reconstruct_fft(c_fft, M):
	"""
	Reconstructs the signal x while given its coefficients.
	Can also be used to extrapolate based on the fft by giving
	an M > len(X).

	c_fft : coefficients of the fft transform of the original signal
	        == output of np.fft.fft(x)
	M : number of samples for which to generate the time-series for.
	"""
	N = len(c_fft)
	phi = 2j * np.pi * np.arange(N)/(N+0.0)
	return np.real(np.exp(np.outer(np.arange(M), phi)).dot(c_fft))/(N+0.0)

def uniformly_sample(x,y, plot=False):
	"""
	Returns samples sampled at equal intervals on the x-axis.
	Uses cubic-spline for interpolation.
	"""
	n,xmin,xmax = len(x), np.min(x), np.max(x)
	x_n = np.linspace(xmin,xmax,n)

	f_y = interpolate.splrep(x,y,s=0)
	y_n = interpolate.splev(x_n,f_y,der=0)
	if plot:
		plt.plot(x,y, label="true")
		plt.plot(x_n,y_n, label="uniformly sampled")
		plt.legend()
		plt.show()
	return x_n,y_n

def get_scaled_x(x0, N):
	alpha, beta = x0[0], x0[1]-x0[0]
	return alpha + beta*np.arange(N)

def predict_CO2():
	"""
	Time series prediction for CO2 data
	"""
	dfname = osp.join(ddir,"co2.mat")
	d = sio.loadmat(dfname)
	t,d = np.squeeze(d['year']), np.squeeze(d['co2'])
	t,d = uniformly_sample(t,d,plot=False)

	N = len(d) # samples
	M = 1.5*len(d) ## number of samples for prediction
	sin_T = 10 # samples -- obtained visually (just need an estimate)
	fs = N+0.0 # set the sampling frequency so that total time = 1.0
	ts = np.arange(N)/(N+0.0) ## time of each sample

	sin_T = (sin_T+0.0)/N 
	sin_w = 1.0/sin_T ## frequency of the sin wave

	d_filt = low_pass(d,fc=0.5*fs, fs=fs) ## cut-off freq = 1/2 sin_w
	
	## now fit an exponential curve to the data:
	## y ~=~ exp(a*x)
	t_filt_good = 0.83 ## time till which there are no edge effects to be seen
	n_good      = int(t_filt_good*N)
	d_fit = d_filt[:n_good]

	c     = d_fit[0]
	d_fit = d_fit - c ## so that d[0]==0.0
	dp, xp = d_fit, np.arange(n_good)

	## polynomial fit:
	## regress on the polynomial basis : 1,x,x^2,x^3,..
	p = np.polyfit(xp,dp,2)
	d_predict = np.polyval(p,np.arange(M)) + c
	
	## fit a pure-tone to the residual:
	d_res = d-d_predict[:N]
	res_fft = np.fft.fft(d_res)
	fft_abs = np.abs(res_fft)
	peaks   = np.arange(len(res_fft))[np.abs(fft_abs-np.max(fft_abs))<1e-2]
	fft_new = np.zeros(len(res_fft), dtype='complex')
	power   =  np.sum(fft_abs)
	fft_new[peaks] = 1.5*res_fft[peaks]

	plt.subplot(211)
	plt.plot(fft_abs)
	plt.stem(np.arange(len(fft_abs))[peaks], np.abs(fft_new)[peaks])
	plt.legend(("FFT of residual", "clean FFT"))
	plt.subplot(212)
	plt.plot(d_res, '--', label="residual")
	plt.plot(reconstruct_fft(fft_new, len(res_fft)), label="clean residual")
	plt.legend()
	plt.show()

	d_predict += reconstruct_fft(fft_new, M)
	plt.plot(ts, d, label="actual")
	plt.plot(get_scaled_x(ts,M), d_predict, label="prediction")
	plt.plot(ts[:n_good], d_fit+c, label="low-passed")
	plt.legend()
	plt.show()

predict_CO2()


