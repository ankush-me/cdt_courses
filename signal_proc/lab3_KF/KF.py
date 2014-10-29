import sys
sys.path.append('/Users/ankushgupta/cdt_courses/signal_proc')

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio

class KF:
	def time_update(self, x, S, A_t, R_t):
		"""
		x,S  : previous belief mean and covariance.
		A_t, R_t : time-step model parameters : x_t = A_t*x_t-1 + e ~ N(0,R_t)

		Returns the new mean and covariance.
		"""
		x_n = A_t.dot(x)
		S_n = A_t.dot(S).dot(A_t.T) + R_t
		return x_n, S_n

	def observation_update(self, x,S, o_t, B_t, Q_t):
		"""
		x,S  : previous belief mean and covariance.
		o_t  : observation
		A_t, R_t : Observation model parameters : y_t = B_t*x_t-1 + e ~ N(0,Q_t)

		Returns the new mean and covariance.
		"""
		eps = np.spacing(0)
		r  = o_t - B_t.dot(x)
		Sr = B_t.dot(S).dot(B_t.T) + Q_t
		K  = S.dot(B_t.T).dot(np.linalg.inv(Sr+eps*np.eye(len(Sr))))
		x_n = x + K.dot(r)

		S_n = (np.eye(len(S)) - K.dot(B_t)).dot(S)
		return x_n, S_n


def load_kf1D():
	d = sio.loadmat('kf1d_signal.mat')
	return np.squeeze(d['t']), np.squeeze(d['true']), np.squeeze(d['noisy'])

def v2d(x):
	return np.atleast_2d(x).T


def AR_predict(d,coeffs, name="", plot=False):
	"""
	Plot the auto-regression predictions and the actual data.
	"""
	c = coeffs
	p = len(c)
	y_predict = np.convolve(d,c[::-1],mode='valid')
	y_predict = y_predict[:-1] ## discard the last value because its outside our domain

	if plot:
		y_gt = d[p:]
		N = len(y_gt)
		plt.subplot(2,1,1)
		ax1 = plt.gca()
		ax1.plot(np.arange(N), y_gt, label="actual")
		ax1.plot(np.arange(N), y_predict, label="prediction %s (p=%d)"%(name,p))
		ax1.legend()
		plt.subplot(2,1,2)
		ax2 = plt.gca()
		ax2.plot(np.arange(p), c[::-1], label= name+' coefficients')
		ax2.legend()
		plt.show()
	return y_predict

def autoregression_kf(p=100):
	"""
	p : number of autor-regression parameters.
	"""
	ts,s,s_n = load_kf1D()

	N = len(ts) ## number of data-points
	T = N-p ## number of kalman steps
	W = np.empty((p, T))  ## store all the sequential-estimates of the weights
	V = np.empty((p, T)) ## store all the variances
	
	A = np.eye(p) # time-step matrix
	R = 1e-7*np.eye(p) # process noise standard-deviation
	Q = v2d(np.array(1e4)) # observation noise variance

	kf = KF()
	w0 = (1./p)*v2d(np.ones(p)) ## initial estimates of the weights
	W[:,0] = np.squeeze(w0)
	S      = 1e-3*np.eye(p)  ## initial variance
	V[:,0] = np.diag(S)

	for t in xrange(T-1):
		B_t = v2d(s[t:p+t]).T
		o_t = v2d(s[p+1+t])
		wT,ST = kf.time_update(v2d(W[:,t]), S, A, R)
		wO, S = kf.observation_update(wT, ST, o_t, B_t, Q)
		W[:,t+1] = np.squeeze(wO)
		V[:,t+1]  = np.diag(S)

	w_final = W[:,-1]
	V = np.sqrt(V)

	s_predict = AR_predict(s, np.squeeze(w_final) ,plot=False)
	plt.subplot(2,1,1)
	plt.plot(ts,s,'g', label="true signal")
	plt.plot(ts[p:],s_predict, 'r', label="predicted signal")
	plt.scatter(ts,s_n, c="g", label="noisy signal", alpha=0.10)
	plt.legend()
	
	plt.subplot(2,1,2)
	plt.plot(W.T)
	for i in xrange(p):
		plt.fill_between(np.arange(T), W[i,:]+V[i,:], W[i,:]-V[i,:], alpha=0.05)

	plt.show()

autoregression_kf(p=25)




