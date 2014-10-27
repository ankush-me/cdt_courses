import numpy as np
import numpy.linalg as nla
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
import datetime as dt

import cov


eps = 1e-8

ddir = "/Users/ankushgupta/cdt_courses/data_estimation/data"
fname= "sotonmet.txt"
d = pd.read_csv(osp.join(ddir, fname), parse_dates=[0,2])
## get the time difference wrt the first entry:
d0 = d[d.columns[0]]
dt = d0 - d0[0]
tot_mins = np.array([t.total_seconds()/60.0 for t in dt])
assert (tot_mins >= 0).all()
colmap = {'t': 4, 't_gt': 9,
 		  'h': 5, 'h_gt': 10}

def dcol(i):
	"""
	returns the i_th column of the DataFrame 'd'
	"""
	return np.array(d[d.columns[i]])

def visualize_pandas():
	t_ser = pd.Series(dcol(colmap['h']).values, dcol(2))
	t_gt_ser= pd.Series(dcol(colmap['h_gt']).values, dcol(2))
	t_gt_ser.plot()
	t_ser.plot(style='.')

def viz():
	##  visualize the data:
	plt.subplot(2,1,1)
	plt.plot(tot_mins, d[d.columns[4]], label=d.columns[4])
	plt.title(d.columns[4])

	plt.subplot(2,1,2)
	plt.plot(d[d.columns[0]], d[d.columns[5]], label=d.columns[5])
	plt.title(d.columns[5])


def make_psd(M, p=0.0, verbose=True):
	"""
	Returns a PSD or PD approximation to the input matrix M
	by setting the non-positive eigen-values to p.
	If M is already PSD, it does not modify it.
	"""
	assert np.allclose(M, M.T), "matrix is not symmetric."
	l,U = nla.eigh(M)
	if (l > 2*eps).all():
		return M

	## ah, it is negative (semi)-definite:
	neg_l = l[l <= 2*eps]
	l[ l <= 2*eps ] = p
	M_psd = U.dot(np.diag(l)).dot(U.T)

	if verbose:
		fig, ax = plt.subplots(1,3)
		plt.subplot(1,3,1)
		plt.imshow(M)
		ax[0].set_title('Original Matrix M')
		
		plt.subplot(1,3,2)
		plt.imshow(M_psd)
		ax[1].set_title('Updated PSD Matrix M_psd')
		
		plt.subplot(1,3,3)
		im = plt.imshow(M_psd-M)
		ax[2].set_title('M_psd - M')
		
		cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
		plt.suptitle('Making PSD using Spectral Decomposition')
		fig.colorbar(im, cax=cbar_ax)
		plt.show(block=False)
		print "There are %d negative eigen-values : "%len(neg_l), neg_l

	return M_psd


class mu_constant:
	"""
	Class to represent constant valued function -- suitable for use as a 
	mean function for the GP.

	@params : 
		c : value of the fucntion
	"""
	def __init__(self, c=0):
		self.c = c

	def get_mu(self, x):
		return self.c*np.ones_like(x)


class GPR:
	"""
	Gaussian Process Regression:
		This class performs gaussian process regression from R^d --> R^m.
		@params:
			f_mu : R^d --> R^m : The mean **function**
			f_k  : R^d x R^d --> R : The covariance kernel **function**
	"""
	def __init__(self, f_mu, f_k):
		self.f_mu = f_mu
		self.f_k  = f_k

	def sample(self, x_nd):
		"""
		Returns a function sampled from the distribution ~N( f_mu(.), f_k(.,.) ) of functions
		sampled at the given n points in R^d
		@return : y_nm : n m-dimensional points and the covariance matrix
		"""
		mu = self.f_mu.get_mu(x_nd)
		S  = self.f_k.get_covmat(x_nd)
		S  = make_psd(S, p=1e-8, verbose=False)
		return np.random.multivariate_normal(mu, S), S


	def predict(self, xi_nd, yi_n, xo_md):
		"""
		Predict y at xo_md given the {xi_nd, yi_n} data-set,
		Returns the mean and co-variance of P(y| xi,yi,xo).
		"""
		if xi_nd.ndim==1: xi_nd = np.atleast_2d(xi_nd).T
		if xo_md.ndim==1: xo_md = np.atleast_2d(xo_md).T
		if yi_n.ndim==1: yi_n   = np.atleast_2d(yi_n).T

		n,d = xi_nd.shape
		m,e = xo_md.shape

		print xi_nd.shape, xo_md.shape
		assert d==e, "GPR.predict : dimension mismatch."

		x_io = np.r_[xi_nd, xo_md]
		K = self.f_k.get_covmat(x_io)
		K_ii, K_io, K_oo = K[:n,:n], K[:n, n:n+m], K[n:n+m, n:n+m]

		Kii_inv = np.linalg.solve(K_ii, np.eye(n))

		mu_i, mu_o = self.f_mu.get_mu(xi_nd), self.f_mu.get_mu(xo_md)
		mu_cond = mu_o + K_io.T.dot(Kii_inv).dot(yi_n-mu_i)
		K_cond  = K_oo - K_io.T.dot(Kii_inv).dot(K_io)

		return mu_cond, K_cond

def plot_gpr(x,y,std, y_gt=None, ax=None):
	"""
	plot (x,y) with std being the standard deviation.
	ax : matplotlib axis
	y_gt : ground truth data
	"""
	if ax==None:
		_,ax = plt.subplots()

	x,y,std = np.squeeze(x), np.squeeze(y), np.squeeze(std)
	ax.fill_between(x, y+std, y-std, alpha=0.2, facecolor='r')
	ax.fill_between(x, y+2*std, y-2*std, alpha=0.2, facecolor='r')
	ax.plot(x,y,'0.40', linewidth=2, label='prediction')
	if y_gt!=None:
		ax.plot(x,y_gt,'g.',label='ground truth')
	plt.legend()
	plt.show()

def test_sample_gpr():
	"""
	Sample the gaussian process and plot it.
	"""
	gpr = GPR(mu_constant(5), cov_se(.1,15))
	x = np.linspace(0,100,1000)
	y, S = gpr.sample(x)
	std  = np.sqrt(np.diag(S))
	plot_gpr(x,y,std)

def test_predict_gpr():
	"""
	Test prediction on the data-set using hand-tuned
	covariance parameters.
	"""
	x = tot_mins
	y = dcol(colmap['h'])
	y_gt = dcol(colmap['h_gt'])


	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]

	f_mu = mu_constant(np.mean(yi_n))
	f_cov= cov.CovSqExpARD()
	f_cov.set_hyperparam(1,np.array([50]), 0.01)

	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

	plot_gpr(x,yo_mu, y_std, y_gt)
	
#visualize_pandas()
test_predict_gpr()
#plt.show()
#test_sample_gpr()
