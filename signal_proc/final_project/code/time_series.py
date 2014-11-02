import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy import interpolate
import os.path as osp
from scipy.signal import butter, filtfilt, find_peaks_cwt, fftconvolve
from auto_regression import *

ddir = "/Users/ankushgupta/cdt_courses/signal_proc/final_project/data"


def low_pass(x,  fc, fs):
    """
    Removes high-frequency content from an array X.
    fs : sampling frequency
    """
    lowcut  = 0.0   # lower-most freq (Hz)
    highcut = fc   # higher-most freq (Hz)

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
	"""
	Scales np.arange(N) to be compatible with x0.
	Useful in plotting future (out-of-data) values.
	"""
	alpha, beta = x0[0], x0[1]-x0[0]
	return alpha + beta*np.arange(N)

def predict_CO2(return_full=False, plot=False):
	"""
	Time series prediction for CO2 data
	"""
	dfname = osp.join(ddir,"co2.mat")
	d = sio.loadmat(dfname)
	t,d = np.squeeze(d['year']), np.squeeze(d['co2'])
	t,d = uniformly_sample(t,d,plot=False)

	N = len(d) # samples
	M = 1.6*len(d) ## number of samples for prediction
	sin_T = 10 # samples -- obtained visually (just need an estimate)
	fs = N+0.0 # set the sampling frequency so that total time = 1.0
	ts = np.arange(N)/(N+0.0) ## time of each sample

	sin_T = (sin_T+0.0)/N 
	sin_w = 1.0/sin_T ## frequency of the sin wave

	d_filt = low_pass(d,fc=10, fs=fs) ## cut-off freq = 1/2 sin_w

	## now fit an exponential curve to the data:
	## y ~=~ exp(a*x)
	t_filt_good = 1.0 ## time till which there are no edge effects to be seen
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

	d_predict += reconstruct_fft(fft_new, M)
		
	if plot:
		plt.subplot(211)
		plt.plot(fft_abs)
		plt.stem(np.arange(len(fft_abs))[peaks], np.abs(fft_new)[peaks])
		plt.legend(("FFT of residual", "clean FFT"), loc=9)
		plt.subplot(212)
		plt.plot(d_res, '--', label="residual")
		plt.plot(reconstruct_fft(fft_new, len(res_fft)), label="clean residual")
		plt.legend()
		plt.suptitle("Residual Analysis")
		plt.show()

		plt.plot(t, d, label="actual")
		plt.plot(get_scaled_x(t,M)[N-10:], d_predict[N-10:], label="prediction")
		plt.plot(t[:n_good], d_fit+c, label="low-passed")
		plt.legend(loc=4)
		plt.xlabel("year")
		plt.ylabel("CO2 levels")
		plt.grid()
		plt.show()

	if return_full:
		return d_predict, d, d_fit+c, p
	else:
		return d_predict, d



def predict_sunspots(p=100):
	"""
	Time series prediction for Sunspots data using linear 
	auto-regressive model.
		p : is the model order.
	"""
	dfname = osp.join(ddir,"sunspots.mat")
	d = sio.loadmat(dfname)
	t,d = np.squeeze(d['year']), np.squeeze(d['activity'])
	d_mu = np.mean(d)
	d = d - d_mu ## center the data
	id_1990 = int(np.nonzero(t==1990)[0])
	t_train, d_train, t_test, d_test = t[:id_1990], d[:id_1990], t[id_1990:], d[id_1990:]

	coeffs_autocorr = AR_lstsq(d,p=p)

	def rolling_regression(dd, c, n_predict=12):
		"""
		Return n_predictions into the future, given
		d entries from the past and auto-regression coeffs c.

		order of enteries:
		d : d[t-p], ...., d[t-1]
		c : c[t-p], ...., c[t-1]
		"""
		yout = np.empty(n_predict)
		y_prev = dd
		for i in xrange(n_predict):
			yout[i] = np.sum(y_prev*c)
			y_prev[:-1] = y_prev[1:]
			y_prev[-1]  = yout[i]
		return yout

	t_pred_idx = 12*np.ceil(p/12.0) ## calculate the index of the first year I can predict.
	t_pred = t[t_pred_idx:]
	d_pred = np.zeros_like(t_pred)
	n_predict_years = len(t_pred)/12 + 1
	for i in xrange(n_predict_years):
		n_predict = min(len(t_pred)-12*(i), 12)
		t_idx = t_pred_idx+i*12
		d_pred[i*12:i*12+n_predict] = rolling_regression(d[t_idx-p:t_idx].copy(), coeffs_autocorr, n_predict)
	

	corr = fftconvolve(d[t_pred_idx:], d_pred[::-1], mode='full')
	i_mid = (len(corr))/2
	i_peak = np.argmax(corr)
	di_step = (i_peak-i_mid)
	print "need to adjust by = %d, p=%d"%(di_step, p)
	di = di_step*(t[1]-t[0])

	plt.plot(corr)
	plt.scatter([i_mid], [corr[i_mid]], c='g', label="mid (ideal peak)")
	plt.scatter([i_peak], [corr[i_peak]], c='r', label="peak")
	plt.title("Cross-Correlation b/w Ground-Truth & Predicted Sunspot Data for Lag Detection")
	plt.legend()
	plt.show()
	
	plt.plot(t_train, d_mu+d_train, 'b-.', label="ground truth (train)")
	plt.plot(t_test,  d_mu+d_test, 'r-.', label="ground truth (test)")
	plt.plot(t_pred,  d_mu+d_pred, 'g', label="prediction")
	plt.title("Sunspot Prediction Using %d-Linear Auto-regressive Model AR(%d)"%(p,p))
	plt.xlabel("year")
	plt.ylabel("activity")
	plt.legend()
	plt.show()


def sweep_p_ss():
	dfname = osp.join(ddir,"sunspots.mat")
	d = sio.loadmat(dfname)
	t,d = np.squeeze(d['year']), np.squeeze(d['activity'])
	ps= np.array([2,5,10,15,20,50,70,100,150,200,250,300,400,500,600])
	err= np.zeros(len(ps))
	ys = []
	for i in xrange(len(ps)):
		p = ps[i]
		coeffs = AR_autocorr(d,p=p)
		y_p, dgt = predict_sunspots(p=p)
		ys.append(y_p)
		err[i] = rms(y_p, dgt)

	plt.plot(ps, np.exp(err), marker='.')
	plt.xlabel('p (auto-regression order)')
	plt.ylabel("rms error")
	plt.title("Sunspots Auto-Regression Error Vs. Model Order")
	plt.show()


predict_CO2(return_full=False, plot=True)
#predict_sunspots(p=15)
#sweep_p_ss()

