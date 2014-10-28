import numpy as np 
import matplotlib.pylab as plt
import scipy.io as sio

d = sio.loadmat('fXSamples.mat')
d = d['x']
d1 = d[:,0] - np.mean(d[:,0])
d2 = d[:,1] - np.mean(d[:,1])
N  = len(d1)


def rms(x1,x2):
	x1,x2 = np.squeeze(x1), np.squeeze(x2)
	ret = np.log(np.linalg.norm(x1-x2)) - np.log(np.sqrt(len(x1)+0.0))
	return ret

def AR_predict(d,coeffs, names=None, plot=False):
	"""
	Plot the auto-regression predictions and the actual data.
	"""
	predictions, ax1, ax2 = [], None, None
	i = 0
	for c in coeffs:
		p = len(c)
		y_predict = np.convolve(d,c[::-1],mode='valid')
		y_predict = y_predict[:-1] ## discard the last value because its outside our domain
		predictions.append(y_predict)

		if plot:
			series_name = names[i] if names!=None else ""
			y_gt = d[p:]
			N = len(y_gt)
			plt.subplot(2,1,1)
			if ax1== None:
				ax1 = plt.gca()
				ax1.plot(np.arange(N), y_gt, label="actual")
			ax1.plot(np.arange(N), y_predict, label="prediction %s (p=%d)"%(series_name,p))
			ax1.legend()
			plt.subplot(2,1,2)
			if ax2==None: ax2 = plt.gca()
			ax2.plot(np.arange(p), c[::-1], label= series_name+' coefficients')
			ax2.legend()
			i += 1
	if plot: plt.show()

	return predictions


def AR_lstsq(d, p):
	"""
	Find auto-regression coefficients through least-squares.
	@param : 
		p : order of the AR model.
	"""
	N = len(d)
	assert p < N
	N = len(d)
	y = d[p:]
	M = np.empty((len(y), p))
	for i in xrange(len(y)):
		M[i,:] = d[i:p+i]

	return np.linalg.lstsq(M,y)[0]
	#return np.linalg.pinv( M.T.dot(M) + 1e-2*np.eye(p)).dot(M.T).dot(y)

def AR_autocorr(d, p):

	def get_autocorr(d, T):
		"""
		Return the auto-correlation matrix for the series d 
		and its T-shifted version.

		It assumes that d is zero-mean.
		"""
		T = np.abs(T)
		N = len(d)-T
		assert T < N

		if T==0: return np.sum(d*d)/(N-1.0)
		else: return np.sum(d[:-T]*d[T:])/(N-1.0)

	auto_corrs = np.array([get_autocorr(d,i) for i in np.arange(p)])
	if p!=1: auto_corrs = np.r_[auto_corrs[:0:-1],auto_corrs]

	M = np.empty((p,p))
	for i in xrange(p): M[i,:] = auto_corrs[i:i+p]

	return np.linalg.inv(M).dot(np.r_[auto_corrs[p:], get_autocorr(d,p)])

def spectrum(coeff, plot=False):
	"""
	coeff : the auto-regression coefficients [a_{p}, a_{p-1},...a_{-1}]
	"""
	p = len(coeff)
	f = np.atleast_2d(np.arange(2000)).T # freq in Hz
	k = np.atleast_2d(np.arange(1,p+1)).T
	a = np.atleast_2d(coeff[::-1]).T
	phi = -2j*np.pi*f.dot(k.T)
	denom = np.abs(1 + np.exp(phi.dot(a)))
	P = 1.0/denom

	if plot:
		plt.plot(f, P, label="Spectrum")
		plt.legend()
		plt.show()
	return P


def AR_crosscorr(d, z, p):
	assert len(d)==len(z)
	def get_crosscorr(d,z, T):
		"""
		Return the auto-correlation matrix for the series d 
		and its T-shifted version.

		It assumes that d is zero-mean.
		"""
		N = len(d)-T
		assert T < N

		if T==0: return np.sum(d*z)/(N-1.0)
		elif T>0: return np.sum(d[:-T]*z[T:])/(N-1.0)
		else: return np.sum(z[:-T]*d[T:])/(N-1.0)

	cross_corrs = np.array([get_crosscorr(d,z,i) for i in np.arange(-p+1,p)])
	M = np.empty((p,p))
	for i in xrange(p): M[i,:] = cross_corrs[i:i+p]

	return np.linalg.inv(M).dot(np.r_[cross_corrs[p:], get_crosscorr(d,z,p)])

def plot_spectrum():
	p = 1000
	coeffs = AR_autocorr(d1,p=p)
	spectrum(coeffs, plot=True )


def test_AR_predict():
	coeffs_lstsq    = AR_lstsq(d1,p=10)
	coeffs_autocorr = AR_autocorr(d1,p=10)
	coeffs_crosscorr = AR_crosscorr(d1,d2,p=10)

	y_p = AR_predict(d1,[coeffs_lstsq, coeffs_autocorr, coeffs_crosscorr],
					 ["lstsq", "autocorr", "crosscorr"], plot=True)
		

def sweep_p():
	ps= np.array([2,5,10,50,100, 200,500,1000])
	err= np.zeros(len(ps))
	for i in xrange(len(ps)):
		p = ps[i]
		coeffs = AR_crosscorr(d1,d2,p=p)
		y_p = AR_predict(d1,[coeffs])
		err[i] = rms(y_p, d1[p:])
	plt.plot(ps, np.exp(err))
	plt.xlabel('p (auto-regression order)')
	plt.ylabel("error")
	plt.show()


#test_AR_predict()
sweep_p()
#test_AR_predict()
#AR_autocorr(d1,10)
#plot_spectrum()