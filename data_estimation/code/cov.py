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

	def print_hyperparam(self):
		print "Square-Exponential Hyperparams:\n==============================="
		print "signal std    : ", np.sqrt(self.a2)
		print "length-scales :", self.b
		print "noise std     : ", np.sqrt(self.c2)


	def nll(self, log_abc, x_nd, y_n, grad=False, use_self_hyper=True):
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
		if use_self_hyper:
			a2,b,c2, = self.a2, self.b, self.c2
		else:
			log_abc = np.squeeze(log_abc)			
			assert len(log_abc) >= 3, "SqExp Cov : Too few hyper-parameters"
			a2 = np.exp(2*log_abc[0])
			b  = np.exp(log_abc[1:-1])
			c2 = np.exp(2*log_abc[-1])

		## make the data-points a 2D matrix:
		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T
		if y_n.ndim==1:
			y_n = np.atleast_2d(y_n).T

		N,d = x_nd.shape
		assert len(y_n)==N, "x and y shape mismatch."
		x_nd = x_nd/b  ## normalize by the length-scale
		D = ssd.pdist(x_nd, 'sqeuclidean')
		K = a2 * np.exp(-0.5*ssd.squareform(D))
		K_n = K + c2*np.eye(N)

		## compute the determinant using LU factorization:
		sign, log_det = nla.slogdet(K_n)
		assert sign > 0, "SqExpCov : covariance matrix is not PSD."
		
		## compute the inverse of the covariance matrix through gaussian
		## elimination:
		K_inv = nla.solve(K_n, np.eye(N))
		Ki_y  = K_inv.dot(y_n)

		## negative log-likelihood:
		nloglik = 0.5*( N*self.log_2pi + log_det + y_n.T.dot(Ki_y))
		
		if grad: ## compute the gradient wrt to hyper-params:
			K_diff = K_inv - Ki_y.dot(Ki_y.T)
			Tr_arg = K_diff * K

			dfX    = np.empty(d+2)
			dfX[0] = np.sum(np.sum(Tr_arg))
			dfX[-1]= c2 * np.trace(K_diff)

			for i in xrange(d):
				xd = np.atleast_2d(x_nd[:,i]).T
				x_sqdiff = ssd.squareform(ssd.pdist(xd, 'sqeuclidean'))
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


	def train(self, th0, x_nd, y_n):
		"""
		Find the optimal value of the hyper-parameters a,b,c
		given the training data of length n.

		Uses scipy's optimization function: scipy.optimize.minimize	
		"""
		if x_nd.ndim==1: x_nd = np.atleast_2d(x_nd).T

		n,d = x_nd.shape
		assert len(th0)==d+2, "cov.train : initial guess shape mismatch"

		print "Optimizing hyper-parameters using conjugate gradient :"
		res = opt.minimize(self.nll, th0, args=(x_nd, y_n,True,False),
						   method='CG', jac=True,
						   options={'maxiter':20, 'disp':False})
		return res


class Periodic:
	"""
	The periodic kernel:

	k(x1,x2) = a^2 * exp(- d^2 * sin^2(2*pi/b *  (x1 - x2)) + c^2*I
		where,
			a : signal standard-dev
			d : slope of decay
			b : time-period of the signal
			c : observation noise standard-dev

	Note: this works for only 1D data-points.
	"""
	def __init__(self):
		self.log_2pi = np.log(2*np.pi)
		self.a2, self.d2, self.b, self.c2 = None,None,None,None

	def _check_params(self):
		assert (self.a2!=None and self.d2!=None
				and self.b!=None and self.c2!=None), "Periodic Cov : null hyperparams."

	def set_hyperparam(self, a,d,b,c):
		"""
		Set the hyperparameters.
		self.{a,d,b,c} are always the parameters themselves (not their logs).
		"""
		self.a2, self.d2, self.b, self.c2 = a*a, d*d, b, c*c
		assert self.b>0

	def set_log_hyperparam(self, lg_a,lg_d,lg_b,lg_c):
		self.set_hyperparam(np.exp(lg_a), np.exp(lg_d), np.exp(lg_b), np.exp(lg_c))

	def print_hyperparam(self):
		print "Periodic Covar Hyperparams:\n==============================="
		print "signal std    : ", np.sqrt(self.a2)
		print "decay-slope   : ", np.sqrt(self.d2)
		print "time-period   : ", self.b
		print "noise std     : ", np.sqrt(self.c2)

	def nll(self, log_adbc, x_nd, y_n, grad=False, use_self_hyper=True):
		"""
		Returns the negative log-likelihood : -log[p(y|x,th)],
		where, abc are the LOG -- hyper-parameters. 
			If adbc==None, then it uses the self.{a,d,b,c}
			to compute the value and the gradient.

		@params:
			x_nd    : input vectors in R^d
			y_n     : output at the input vectors
			log_adbc : vector of hyperparameters:
					 = [log[a], log[d], log[b], log[c]] 
			grad : if TRUE, this function also returns
				   the partial derivatives of nll w.r.t
				   each (log) hyper-parameter.
		"""
		if use_self_hyper:
			a2,d2,b,c2, = self.a2, self.d2, self.b, self.c2
		else:
			log_adbc = np.squeeze(log_adbc)
			self.set_log_hyperparam(log_adbc[0],log_adbc[1],log_adbc[2],log_adbc[3])
			assert len(log_adbc) == 4, "Periodic Cov : incorrect number of params."
			a2 = np.exp(2*log_adbc[0])
			d2 = np.exp(2*log_adbc[1])
			b  = np.exp(log_adbc[2])
			c2 = np.exp(2*log_adbc[3])


		## make the data-points a 2D matrix:
		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T
		if y_n.ndim==1:
			y_n = np.atleast_2d(y_n).T

		N,d = x_nd.shape
		assert len(y_n)==N, "x and y shape mismatch."

		D = ssd.squareform(ssd.pdist(x_nd, 'cityblock'))
		ang   = 2*np.pi/self.b * D
		sinD  = np.sin(2*np.pi/self.b * D)
		sin2D = sinD**2
		K   = self.a2 * np.exp(-self.d2* sin2D)
		K_n = K + self.c2*np.eye(N)

		## compute the determinant using LU factorization:
		sign, log_det = nla.slogdet(K_n)
		assert sign > 0, "Periodic Cov : covariance matrix is not PSD."
		
		## compute the inverse of the covariance matrix through gaussian
		## elimination:
		K_inv = nla.solve(K_n, np.eye(N))
		Ki_y  = K_inv.dot(y_n)

		## negative log-likelihood:
		nloglik = 0.5*( N*self.log_2pi + log_det + y_n.T.dot(Ki_y))
		
		if grad: ## compute the gradient wrt to hyper-params:
			K_diff = K_inv - Ki_y.dot(Ki_y.T)
			Tr_arg = K_diff * K

			dfX    = np.zeros(4)
			dfX[0] = np.sum(np.sum(Tr_arg))
			dfX[1] = np.sum(np.sum(-np.sqrt(d2)*sin2D*Tr_arg))
			cosD   = np.cos(2*np.pi/self.b * D)
			dfX[2] = np.sum(np.sum(Tr_arg*self.d2*sinD*cosD*ang))
			dfX[3] = c2 * np.trace(K_diff)
			return nloglik, dfX

		return nloglik


	def get_covmat(self, x_nd, check_psd=False):
		"""
		Returns an nxn PSD covariance matrix as defined by the covariance function f_k
		at the given input points.

			k(x1,x2) = a^2 * exp(- d^2 * sin^2(2*pi/b *  (x1 - x2)) + c^2*I

		@params:
			x_nd   : n d-dimensional row-vectors (input points).
			check_psd : sanity check if the output matrix is PSD.
		"""
		self._check_params()

		## make the data-points a 2D matrix:
		if x_nd.ndim==1:
			x_nd = np.atleast_2d(x_nd).T

		N,d = x_nd.shape
		D = ssd.squareform(ssd.pdist(x_nd, 'cityblock'))
		sin2D = np.sin(2*np.pi/self.b * D)**2
		K = self.a2 * np.exp(-self.d2* sin2D) + self.c2*np.eye(N)

		if check_psd:
			assert (nla.eig(K)[0] >= 0).all(), "Covariance is not PSD."
		return K


	def train(self, th0, x_nd, y_n):
		"""
		Find the optimal value of the hyper-parameters a,b,c
		given the training data of length n.

		Uses scipy's optimization function: scipy.optimize.minimize	
		"""
		if x_nd.ndim==1: x_nd = np.atleast_2d(x_nd).T

		n,d = x_nd.shape
		assert len(th0)==4, "cov.train : initial guess shape mismatch"

		print "Optimizing hyper-parameters using conjugate gradient :"
		res = opt.minimize(self.nll, th0, args=(x_nd, y_n,True,False),
						   method='CG', jac=True,
						   options={'maxiter':15, 'disp':False})
		return res

