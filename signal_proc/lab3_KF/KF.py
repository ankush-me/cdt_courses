import sys
sys.path.append('/Users/ankushgupta/cdt_courses/signal_proc')

import time
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
		K  = S.dot(B_t.T).dot(np.linalg.inv(Sr))
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
		B_t = v2d(s_n[t:p+t]).T
		o_t = v2d(s_n[p+1+t])
		wT,ST = kf.time_update(v2d(W[:,t]), S, A, R)
		wO, S = kf.observation_update(wT, ST, o_t, B_t, Q)
		W[:,t+1] = np.squeeze(wO)
		V[:,t+1]  = np.diag(S)

	w_final = W[:,-1]
	V = np.sqrt(V)

	s_predict = AR_predict(s_n, np.squeeze(w_final) ,plot=False)
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

def boat_racing():
	d = sio.loadmat('kf2d_signal.mat')
	t_n,x_n,y_n = np.squeeze(d['t_n']), np.squeeze(d['x_n']), np.squeeze(d['y_n'])
	sort_tn = np.argsort(t_n)
	t_n, x_n, y_n = t_n[sort_tn], x_n[sort_tn], y_n[sort_tn]
	ts ,x_gt ,y_gt = np.squeeze(d['t']), np.squeeze(d['x']), np.squeeze(d['y'])
	T = len(ts)
	

	def const_xy():
		dt = 1.0
		A  = np.eye(2)
		B  = np.eye(2)
		s_xy = 1.0
		R = np.diag([s_xy, s_xy]) ## process noise covariance
		Q = 300*np.eye(2) ## observation covariance

		x0 = np.zeros(2) # initial estimate
		S0 = 1e-4*np.eye(2) ## initial variance
		return A,B,R,Q,x0,S0	

	def const_velocity():
		dt = 1.0
		A  = np.array([[1.0, 0.0, dt, 0.0],  ## process matrix
				 	   [0.0, 1.0, 0.0, dt],
				 	   [0.0, 0.0, 1.0, 0.0],
				 	   [0.0, 0.0, 0.0, 1.0]])
		B = np.array([[1.0, 0.0, 0.0, 0.0], ## observation matrix
					  [0.0, 1.0, 0.0, 0.0]])
		
		s_xy, s_v = 1e-8, 1e-4
		R = np.diag([s_xy, s_xy, s_v, s_v]) ## process noise covariance
		Q = 30*np.eye(2) ## observation covariance

		x0 = np.zeros(4) # initial estimate
		S0 = 1e-4*np.eye(4) ## initial variance
		return A,B,R,Q, x0, S0		

	def const_accel():
		dt = 1.0
		A  = np.array([[1.0, 0.0,  dt,  0.0,  (dt*dt)/2.0,             0.0],  ## process matrix
				 	   [0.0, 1.0, 0.0,   dt,          0.0,     (dt*dt)/2.0],
				 	   [0.0, 0.0, 1.0,  0.0,           dt,             0.0],
				 	   [0.0, 0.0, 0.0,  1.0,          0.0,              dt],
				 	   [0.0, 0.0, 0.0,  0.0,          1.0,             0.0],
				 	   [0.0, 0.0, 0.0,  0.0,          0.0,             1.0]])

		B = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], ## observation matrix
					  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

		s_xy, s_v, s_acc = 1e-12, 1e-12, 1e-8
		R = np.diag([1e-8, 1e-12, 1e-8, 1e-8, 1e-6, 1e-8]) ## process noise covariance
		Q = 30*np.eye(2) ## observation covariance
		x0 = np.zeros(6) # initial estimate
		S0 = np.diag([1e-4,1e-4,1e-2,1e-2,100,100]) ## initial variance
		return A,B,R,Q, x0, S0		

	A,B,R,Q,x0,S0 = const_velocity()
	n = len(A)
	XY = np.empty((n,T)) ## store all the estimates
	V  = np.empty((n,T)) ## store all the variances

	XY[:,0] = x0
	V[:,0]  = np.diag(S0)
	kf = KF()
	tn_idx = 0

	for t in ts[:-1]:
		x, S = kf.time_update(v2d(XY[:,t]), S0, A, R)
		if tn_idx < len(t_n) and t_n[tn_idx]==t: ## we have an observation:
			o_t  = v2d(np.array([x_n[tn_idx], y_n[tn_idx]]))
			x, S = kf.observation_update(x, S, o_t, B, Q)
			tn_idx += 1

		XY[:,t+1], S0 = np.squeeze(x), S
		V[:,t+1] = np.diag(S0)


	plt.scatter(x_n, y_n, c=t_n, edgecolors='none', label="observation")
	plt.plot(x_gt,y_gt, label="ground-truth")
	plt.plot(XY[0,:], XY[1,:], label="prediction")# c=ts, edgecolor='none', label="prediction")
	plt.legend()
	#plt.colorbar()
	plt.show()
	

#autoregression_kf(p=25)
#boat_racing()



