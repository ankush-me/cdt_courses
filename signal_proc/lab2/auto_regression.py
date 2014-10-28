import numpy as np 
import matplotlib.pylab as plt
import scipy.io as sio

d = sio.loadmat('fXSamples.mat')
d = d['x']
d1 = d[:,0]
d1 = d1 - np.mean(d1)
N  = len(d1)


def rms(x1,x2):
	x1,x2 = np.squeeze(x1), np.squeeze(x2)
	return np.linalg.norm(x1-x2)/np.sqrt(N)

def AR_predict(d,coeffs, names=None, plot=False):
	"""
	Plot the auto-regression predictions and the actual data.
	"""
	predictions, ax1, ax2 = [], None, None
	i = 0
	for c in coeffs:
		p = len(c)
		y_predict = np.convolve(d,c[::-1],mode='valid')
		predictions.append(y_predict)
		if plot:
			series_name = names[i] if names!=None else ""
			y_gt = d[p-1:]
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
	if p!=0: auto_corrs = np.r_[auto_corrs[:0:-1],auto_corrs]
	
	M = np.empty((p,p))
	for i in xrange(p):
		M[i,:] = auto_corrs[i:i+p]
	
	return np.linalg.inv(M).dot(np.r_[auto_corrs[p:], get_autocorr(d,p)])


def test_AR_predict():
	coeffs_lstsq    = AR_lstsq(d1,p=1000)
	coeffs_autocorr = AR_autocorr(d1,p=1000)

	y_p = AR_predict(d1,[coeffs_lstsq, coeffs_autocorr], ["lstsq", "autocorr"], plot=True)
		

def sweep_p():
	ps= np.array([1,2,5,10,50,100])
	err= np.zeros_like(ps)
	for i in xrange(len(ps)):
		p = ps[i]
		coeffs = AR_lstsq(d1,p=p)
		y_p = AR_predict(d1,coeffs)
		err[i] = rms(y_p, d1[p-1:])
	plt.plot(ps, err)
	plt.show()


test_AR_predict()
#AR_autocorr(d1,10)
