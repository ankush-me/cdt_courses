import numpy as np
import numpy.linalg as nla
import scipy.spatial.distance as ssd
import scipy.optimize as opt


class CovSqExpARD:
	"""
	The squared exponential covariance function with
	Automatic Relevance Determination (ARD).

	Works for general n-dimensional data points.

	k(x1,x2) = a^2 * exp(-1/2*|| (x1 - x2)./b||^2) + c^2*I
		where,
			a : signal standard-dev
			b : dimensionwise unit length
			c : observation noise standard-dev
	"""
	def __init__(self):
		self.log_2pi = np.log(2*np.pi)
		self.a2, self.b, self.c2 = None,None,None

	def _check_params(self):
		assert (self.a2!=None and self.b!=None and self.c2!=None), "SqExp Cov : null hyperparams."

	def set_hyperparam(self, a,b,c):
		"""
		Set the hyperparameters.
		self.{a,b,c} are always the parameters themselves (not their logs).
		"""
		self.a2, self.b, self.c2 = a*a,b,c*c
		assert self.b.ndim==1 and (self.b>0).all()

	def set_log_hyperparam(self, lg_a,lg_b,lg_c):
		self.set_hyperparam(np.exp(lg_a), np.exp(lg_b), np.exp(lg_c))


	def nll(self, x_nd, y_n, log_abc=None, grad=False):
		"""
		Returns the negative log-likelihood : -log[p(y|x,th)],
		where, abc are the LOG -- hyper-parameters. 
			If abc==None, then it uses the self.{a,b,c}
			to compute the value and the gradient.

		@params:
			x_nd    : input vectors in R^d
			y_n     : output at the input vectors
			log_abc : vector of hyperparameters:
					 = [log[a], log[b_1],..,log[b_d], log[c]] 
			grad : if TRUE, this function also returns
				   the partial derivatives of nll w.r.t
				   each (log) hyper-parameter.
		"""
		if log_abc == None:
			a2,b,c2, = self.a2, self.b, self.c2
		else:
			assert len(log_abc) >= 3, "SqExp Cov : Too few hyper-parameters"
			a2 = np.exp(2*log_abc[0])
			b  = np.exp(log_abc[1:-1])
			c2 = np.exp(2*log_abc[-1])

		## make the data-points a 2D matrix:
		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T

		N,d = self.x_nd.shape
		assert len(y_n)==n, "x and y shape mismatch."
		x_nd = x_nd/b  ## normalize by the length-scale
		D = ssd.pdist(x_nd, 'sqeuclidean')
		K = a2 * np.exp(-0.5*ssd.squareform(D))
		K_n = K + c2*np.eye(N)

		## compute the determinant using LU factorization:
		sign, log_det = nla.slogdet(K_n)
		assert sign > 0, "SqExpCov : covariance matrix is not PSD."
		
		## compute the inverse of the covariance matrix through gaussian
		## elimination:
		K_inv = np.linalg.solve(K_n, np.eye(N))
		Ki_y  = K_inv.dot(y_n)

		## negative log-likelihood:
		nloglik = 0.5*( N*self.log_2pi + log_det + y_nd.T.dot(Ki_y))

		if grad: ## compute the gradient wrt to hyper-params:
			K_diff = K_i - Ki_y.dot(Ki_y.T)
			Tr_arg = K_diff * K

			dfX    = np.empty(d+2)
			dfX[0] = np.sum(np.sum(Tr_arg))
			dfX[-1]= c2 * np.trace(K_diff)

			for i in xrange(d):
				x_sqdiff = ssd.squareform(ssd.pdist(x_nd[:,i], 'sqeuclidean'))
				dfX[i+1] = 0.5*np.sum(np.sum(Tr_arg * x_sqdiff))

			return nloglik, dfX

		return nloglik


	def get_covmat(self, x_nd, check_psd=False):
		"""
		Returns an nxn PSD covariance matrix as defined by the covariance function f_k
		at the given input points.

		@params:
			x_nd   : n d-dimensional row-vectors (input points).
			check_psd : sanity check if the output matrix is PSD.
		"""
		self._check_params()

		## make the data-points a 2D matrix:
		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T

		N,d = x_nd.shape
		x_nd = x_nd/self.b  ## normalize by the length-scale
		D = ssd.pdist(x_nd, 'sqeuclidean')
		K = self.a2 * np.exp(-0.5*ssd.squareform(D)) + self.c2*np.eye(N)

		if check_psd:
			assert (nla.eig(K)[0] >= 0).all(), "Covariance is not PSD."
		return K


def train(self, f_k, x_nd, y_n):
	"""
	Find the optimal value of the hyper-parameters a,b,c
	given the training data of length n.

	Uses scipy's optimization function: scipy.optimize.minimize	
	"""
	pass


