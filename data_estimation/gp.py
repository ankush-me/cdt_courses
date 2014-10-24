import numpy as np
import numpy.linalg as nla
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
import datetime as dt


eps = 1e-8

ddir = "/Users/ankushgupta/cdt_courses/data_estimation/lab1"
fname= "sotonmet.txt"

"""
d = pd.read_csv(osp.join(ddir, fname), parse_dates=[0,2])
## get the time difference wrt the first entry:
d0 = d[d.columns[0]]
dt = d0 - d0[0]
tot_mins = np.array([t.total_seconds()/60.0 for t in dt])
assert (tot_mins >= 0).all()

colmap = {'t': 4, 't_gt': 9,
 		  'h': 5, 'h_gt': 10}
"""

def dcol(i):
	"""
	returns the i_th column of the DataFrame 'd'
	"""
	return d[d.columns[i]]

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

class cov_se:
	"""
	The squared exponential covariance function.
	Works for general n-dimensional data points.

	f(x1,x2) = a^2 * exp(-b^2 * || x1 - x2||^2)

	NOTE: The general form of the exp argument is : (x1-x2)^T * M * (x1-x2)
	But if M = sI (isotropic-diagonal), then this is equivalent to having s*||(x1-x2)||^2
	"""
	def __init__(self, a=None, b_inv=None):
		"""
		a,b_inv = 1/b as defined above.

		a : Scale of the variance (std^2)
		b : Inverse unit length (1/length units)

		If any of the parameters are set to None,
		this class's .train method can be used to 
		find an optimal value for these hyper-parameters.
		"""
		self.a = a
		self.b = 1.0/(b_inv+eps)

		self.a2 = a*a
		self.b2 = self.b*self.b

	def _check_params(self):
		if self.a == None and self.b == None :
			print "The hyper-parameters are None. Call .train before using."

	def train(self, x_nd, y_nm):
		"""
		Find the optimal value of the hyper-parameters a,b 
		given the training data of length n.
		"""
		pass

	def cov(x1, x2):
		"""
		Returns the covariance for two datapoints.
		"""
		self._check_params()
		return self.a2 * np.exp(-self.b2 * nla.norm(x1-x2))

	def get_covmat(self, x_nd, square=True, check_psd=False):
		"""
		Returns an nxn PSD covariance matrix as defined by the covariance function f_k
		at the given input points.

		@params:
			x_nd   : n d-dimensional row-vectors (input points).
			square : return a square nxn matrix if true, else return n(n-1)/2 pairwise distances.
			check_psd : sanity check if the output matrix is PSD.
		"""
		self._check_params()

		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T

		dists = ssd.pdist(x_nd, 'sqeuclidean')
		if square: 
			K = self.a2 * np.exp(-self.b2 * ssd.squareform(dists))
			if check_psd:
				assert (nla.eig(K)[0] >= 0).all(), "Covariance is not PSD."
			return K
		else: return self.a2 * np.exp(-self.b2 * dists)

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
		S = make_psd(S, p=1e-8, verbose=False)
		return np.random.multivariate_normal(mu, S), S

def plot_gpr(x,y,std, ax=None):
	"""
	plot (x,y) with std being the standard deviation.
	ax : matplotlib axis
	"""
	if ax==None:
		_,ax = plt.subplots()
	ax.fill_between(x, y+std, y-std, alpha=0.2, facecolor='r')
	ax.fill_between(x, y+2*std, y-2*std, alpha=0.2, facecolor='r')
	ax.plot(x,y)
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

#visualize_pandas()
test_sample_gpr()