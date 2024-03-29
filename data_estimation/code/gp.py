import numpy as np
import numpy.linalg as nla
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
import datetime as dt

import cov

## de-clutter terminal output:
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

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

def viz():
	##  visualize the data:
	plt.subplot(2,1,1)
	plt.plot(tot_mins, d[d.columns[4]], label=d.columns[4])
	plt.title(d.columns[4])

	plt.subplot(2,1,2)
	plt.plot(d[d.columns[0]], d[d.columns[5]], label=d.columns[5])
	plt.title(d.columns[5])

def rms(x,x_p):
	x,x_p = np.squeeze(x), np.squeeze(x_p)
	return nla.norm(x-x_p)/np.sqrt(len(x))

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
		assert d==e, "GPR.predict : dimension mismatch."

		x_io = np.r_[xi_nd, xo_md]
		K = self.f_k.get_covmat(x_io)
		K_ii, K_io, K_oo = K[:n,:n], K[:n, n:n+m], K[n:n+m, n:n+m]

		Kii_inv = np.linalg.solve(K_ii, np.eye(n))

		mu_i, mu_o = self.f_mu.get_mu(xi_nd), self.f_mu.get_mu(xo_md)
		mu_cond = mu_o + K_io.T.dot(Kii_inv).dot(yi_n-mu_i)
		K_cond  = K_oo - K_io.T.dot(Kii_inv).dot(K_io)

		return mu_cond, K_cond

def plot_gpr(x,y,std, y_gt=None, xlabel=None, ylabel=None, title=None, ax=None, y_obs=None):
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
	if y_gt!=None: ax.plot(x[:len(y_gt)],y_gt,'g.',label='ground truth')
	if y_obs!=None: ax.plot(x, y_obs, 'r.', label='observation')
	if xlabel!=None: plt.xlabel(xlabel)
	if ylabel!=None: plt.ylabel(ylabel)
	if title!=None : plt.title(title)
	plt.legend()
	plt.show()

def test_sample_gpr():
	"""
	Sample the gaussian process and plot it.
	"""
	gpr = GPR(cov.mu_constant(5), cov_se(.1,15))
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
	y = dcol(colmap['t'])
	y_gt = dcol(colmap['t_gt'])

	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]

	yi_gt = y_gt[d_idx]
	obs_std= np.std(yi_gt-yi_n)
	
	f_mu = cov.mu_constant(np.mean(yi_n))
	f_cov= cov.CovSqExpARD()
	f_cov.set_hyperparam(1,np.array([100]), obs_std)

	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

	plot_gpr(x,yo_mu, y_std, y_gt)
	
def test_train_gpr():
	"""
	Test training GPR covariance hyperparameters.
	"""
	x = tot_mins
	y = dcol(colmap['t'])
	y_gt = dcol(colmap['t_gt'])

	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]
	
	f_cov= cov.CovSqExpARD()
	f_mu = cov.mu_constant(np.mean(yi_n))

	## optimize for the hyper-parameters:
	##   initial guess:
	## t_gt : 2,100,1.0
	signal_std = 2.0
	len_scales = np.array([100])
	obs_std    = 1.0 #0.5
	th0 = np.log(np.r_[signal_std, len_scales, obs_std])
	res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	resx = np.squeeze(res.x)
	print "inital    hyperparams : ", th0
	print "optimized hyperparams : ", resx

	f_cov.set_log_hyperparam(resx[0], resx[1:-1], resx[-1])
	f_cov.print_hyperparam()

	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

	print "** rms error: ", rms(y_gt, yo_mu)

	plot_gpr(x,yo_mu, y_std, y_gt)

def test_train_gpr_periodic():
	"""
	Test training GPR covariance hyperparameters.
	"""
	x = tot_mins
	y = dcol(colmap['h'])
	y_gt = dcol(colmap['h_gt'])

	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]
	
	f_cov= cov.Periodic()
	f_mu = cov.mu_constant(np.mean(yi_n))

	## optimize for the hyper-parameters:
	##   initial guess:
	## t_gt : 2,100,1.0
	signal_std = 1.0
	slope      = 1.0
	period     = 1400
	obs_std    = 1.0
	th0 = np.log(np.r_[signal_std, slope, period, obs_std])
	
	res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	resx = np.squeeze(res.x)
	print "inital    hyperparams : ", th0
	print "optimized hyperparams : ", resx

	f_cov.set_log_hyperparam(resx[0], resx[1], resx[2], resx[3])
	f_cov.print_hyperparam()

	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

	print "** RMS ERROR: ", rms(y_gt, yo_mu)
	plot_gpr(x,yo_mu, y_std, y_gt)


def seq_predict():
	x = tot_mins
	y = dcol(colmap['t'])
	y_gt = dcol(colmap['t_gt'])
	d_idx = np.nonzero(np.isfinite(y))[0]
	
	f_cov= cov.CovSqExpARD()

	# parameters for temperature:
	f_cov.set_hyperparam(2.676621, np.array([65.65937]), 0.32133)
	
	## parameter for height:
	#f_cov.set_hyperparam(0.808839, np.array([88.27811]), 0.0293146)
	f_mu = cov.mu_constant(np.mean(y[d_idx]))
	gpr = GPR(f_mu, f_cov)
	
	#d_idx = d_idx[np.logical_or(d_idx<5, d_idx >=20)]

	M = len(x)
	y_seq_mu = np.empty(M)
	y_seq_std= np.empty(M)
	for i in xrange(M):
		print i
		x_new = x[i]
		x_obs, y_obs = x[d_idx[d_idx<=i]], y[d_idx[d_idx<=i]]
		y_seq_mu[i], var = gpr.predict(x_obs, y_obs, np.atleast_1d(x_new))
		y_seq_std[i] = np.sqrt(var[-1,-1])

	plot_gpr(x,y_seq_mu, y_seq_std, y_gt,
			 xlabel="Time (min)", ylabel="Temperature (C)",
			 title="Sequential Prediction",
			 y_obs=y)
	plt.show(block=True)

def plot_nll():
	x = tot_mins
	y = dcol(colmap['h'])
	y_gt = dcol(colmap['h_gt'])
	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]

	f_cov= cov.CovSqExpARD()
	f_mu = cov.mu_constant(np.mean(yi_n))
	signal_std = 0.80884

	N = 50
	ll,nn = np.meshgrid(np.linspace(np.log(0.1), np.log(150), N),
						np.linspace(np.log(0.005), np.log(5), N))

	z_nll = np.empty((N,N))
	i = 0
	for xi in range(N):
		for yi in range(N):
			i +=1
			print i
			z_nll[yi,xi] =  f_cov.nll(np.r_[np.log(signal_std), ll[yi,xi], nn[yi,xi]], xi_n, yi_n, grad=False, use_self_hyper=False)

	#min_xy = np.unravel_index(np.argmin(z_nll), z_nll.shape)
	cset = plt.contour(ll,nn,z_nll, colors='k')
	ax = plt.gca()
	#xloc, yloc = ll[min_xy[0],min_xy[1]], nn[min_xy[0],min_xy[1]]
	xloc, yloc = np.log(88.27811177), np.log(0.02931)
	ax.scatter(xloc, yloc, s=100, c='r')
	ax.annotate("(88.28, 0.029)", (xloc, yloc),  horizontalalignment='right', color='w')
	im   = plt.imshow(z_nll, origin='lower', cmap=plt.get_cmap('jet'), vmin=z_nll.min(), vmax=z_nll.max(), extent=[np.min(ll), np.max(ll), np.min(nn), np.max(nn)])
	im.set_interpolation('bilinear')
	plt.xlabel("log(length-scale)")
	plt.ylabel("log(observation noise std-dev)")
	plt.title("Negative Log-Likelihood")
	plt.colorbar()
	plt.show()

def generate_plots():
	x = tot_mins
	y = dcol(colmap['t'])
	y_gt = dcol(colmap['t_gt'])
	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]
	N = len(x)
	dx = (np.max(x)-np.min(x))/(N+0.0)
	x_extended  = np.r_[x , np.max(x) + 1 + np.arange(250)*dx]

	"""
	f_cov= cov.Periodic()
	f_mu = cov.mu_constant(np.mean(yi_n))
	signal_std = 1.0
	slope      = 1.0
	period     = 1500
	obs_std    = 1.0
	th0 = np.log(np.r_[signal_std, slope, period, obs_std])
	res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	resx = np.squeeze(res.x)
	rest = resx
	f_cov.set_log_hyperparam(resx[0], resx[1], resx[2], resx[3])
	f_cov.print_hyperparam()
	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x_extended)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T
	print "** {Periodic, Constant} RMS ERROR: ", rms(y_gt, yo_mu[:N])
	plot_gpr(x_extended,yo_mu, y_std, y_gt,
			 xlabel="Time (min)", ylabel="Tide Height (m)",
			 title="COV : Periodic, MEAN : mean(y), rms = %0.3f"%rms(y_gt, yo_mu[:N]))
	plt.show(block=True)
	"""
	
	f_cov= cov.CovSqExpARD()
	f_mu = cov.mu_constant(np.mean(yi_n))
	signal_std = 0.5
	len_scales = np.array([120])
	obs_std    = 1.0
	th0 = np.log(np.r_[signal_std, len_scales, obs_std])
	res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	resx = np.squeeze(res.x)
	f_cov.set_log_hyperparam(resx[0], resx[1:-1], resx[-1])
	f_cov.print_hyperparam()
	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x_extended)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T
	print "*** {SqExp, constant} RMS error: ", rms(y_gt, yo_mu[:N])
	plot_gpr(x_extended,yo_mu, y_std, y_gt,
			 xlabel="Time (min)", ylabel="Tide Height (m)",
			 title="COV : SqExp, MEAN : mean(y), rms = %0.3f"%rms(y_gt, yo_mu[:N]))
	plt.show(block=True)

	"""
	f_cov= cov.Periodic()
	f_mu = cov.mu_spline(xi_n, yi_n)
	signal_std = 1.0
	slope      = 1.0
	period     = 100
	obs_std    = 1.0
	th0 = np.log(np.r_[signal_std, slope, period, obs_std])
	#res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	#resx = np.squeeze(res.x)
	resx = rest
	f_cov.set_log_hyperparam(resx[0], resx[1], resx[2], resx[3])
	f_cov.print_hyperparam()
	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T
	print "** {Periodic, Spline} RMS ERROR: ", rms(y_gt, yo_mu[:N])
	plot_gpr(x,yo_mu, y_std, y_gt,
			 xlabel="Time (min)", ylabel="Tide Height (m)",
			 title="COV : Periodic, MEAN : spline, rms = %0.3f"%rms(y_gt, yo_mu[:N]))
	plt.show(block=True)

	f_cov= cov.CovSqExpARD()
	f_mu = cov.mu_spline(xi_n, yi_n)
	signal_std = 0.1
	len_scales = np.array([1])
	obs_std    = 2
	th0 = np.log(np.r_[signal_std, len_scales, obs_std])
	#res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	#resx = np.squeeze(res.x)
	print "inital    hyperparams : ", th0
	print "optimized hyperparams : ", resx
	f_cov.set_log_hyperparam(resx[0], resx[1:-1], resx[-1])
	f_cov.print_hyperparam()
	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T
	print "** {SqExp, Spline} RMS error: ", rms(y_gt, yo_mu[:N])
	plot_gpr(x,yo_mu, y_std, y_gt,
			 xlabel="Time (min)", ylabel="Tide Height (m)",
			 title="COV : SqExp, MEAN : spline, rms = %0.3f"%rms(y_gt, yo_mu[:N]))
	plt.show(block=True)
	"""


def test_train_gpr_product():
	"""
	Test training GPR covariance hyperparameters.
	"""
	x = tot_mins
	y = dcol(colmap['h'])
	y_gt = dcol(colmap['h_gt'])

	d_idx = np.isfinite(y)
	xi_n, yi_n = x[d_idx], y[d_idx]
	x0_m       = x[np.logical_not(d_idx)]
	
	f_cov= cov.ProductCov(cov.CovSqExpARD(), cov.Periodic(), 3, 4)
	f_mu = cov.mu_constant(np.mean(yi_n))

	th0 = np.log(np.r_[1, 500, 0.4, 1.0, 1.0, 1340, 0.1])
	
	"""
	_, deriv =  f_cov.nll(th0, xi_n, yi_n-f_mu.get_mu(xi_n), True, False)
	df = []
	for i in xrange(len(th0)):
		p = th0[i]
		pp=p+eps
		pn=p-eps
		th1 = th0.copy()
		th1[i] = pp
		fp = f_cov.nll(th1, xi_n, yi_n-f_mu.get_mu(xi_n), False, False)
		th1[i] = pn
		fn = f_cov.nll(th1, xi_n, yi_n-f_mu.get_mu(xi_n), False, False)
		df.append((fp-fn)/(2*eps))
	print "numerical :", df
	print "analytical:", deriv
	"""
	res  = f_cov.train(th0, xi_n, yi_n-f_mu.get_mu(xi_n)) 
	resx = np.squeeze(res.x)
	print "inital    hyperparams : ", th0
	print "optimized hyperparams : ", resx

	f_cov.set_log_hyperparam(resx)
	f_cov.print_hyperparam()

	gpr = GPR(f_mu, f_cov)
	yo_mu, yo_S = gpr.predict(xi_n, yi_n, x)
	y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

	print "** rms error: ", rms(y_gt, yo_mu)
	plot_gpr(x,yo_mu, y_std, y_gt)


### Sanity check : test sampling from a GP:
#test_sample_gpr()

### Predict the output for temperature-data using hand-tuned hyper-params:
#test_predict_gpr()

### Predict the out for temperature-data but this time optimizing the
### hyper-params using marginal likelihood:
#test_train_gpr()

### Train the hyper-parameters for periodic covariance on the tide-data:
#test_train_gpr_periodic()

### Plot the negative-log-likelihood figure used in the report:
#plot_nll()

### Do sequential predictions:
#seq_predict()

### convenience function for generating various plots:
#generate_plots()
