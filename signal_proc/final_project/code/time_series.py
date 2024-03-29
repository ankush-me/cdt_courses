import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy import interpolate
import os.path as osp
from scipy.signal import butter, filtfilt, find_peaks_cwt, fftconvolve

from auto_regression import *
import cov, gp ## GPR code
import nn ## neural-net code
import time, math, sys
import cPickle as cp

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
        print "c: ",c
        return d_predict, d, d_fit+c, p
    else:
        return d_predict, d

class mu_poly:
    """
    Class for returning the mean for a G.P. defined by
    a polynomial basis : mu(x) = a0 + a1*x + a2*x^2 + ...

    @params:
        p : coefficients of the polynomial basis : higest degree first.
    """
    def __init__(self, p):
        self.p = p

    def get_mu(self,x):
        return np.polyval(self.p, x)

def predict_CO2_gp(plot=True):
    """
    Predict CO2 data using periodic covariance Gaussian Process Regression.
    """
    dfname = osp.join(ddir,"co2.mat")
    d      = sio.loadmat(dfname)
    t,d = np.squeeze(d['year']), np.squeeze(d['co2'])
    t,d = uniformly_sample(t,d,plot=False)

    _,_,d_fit,p = predict_CO2(return_full=True, plot=False)
    c = d_fit[0]

    d = d-c
    f_mu = mu_poly(p)

    """
    ## for square-exponential:
    f_cov= cov.CovSqExpARD()
    signal_std = 0.5
    len_scales = np.array([10])
    obs_std    = 0.02 
    th0 = np.log(np.r_[signal_std, len_scales, obs_std])
    """
    ## optimize for the hyper-parameters:
    ##   initial guess:
    f_cov= cov.Periodic()
    signal_std = 2.53303896031
    slope      = 34.6180235039
    period     = 527.686276333
    obs_std    = 1.0
    
    th0 = np.log(np.r_[signal_std, slope, period, obs_std])
    xs = np.arange(len(d))
    ys = d - f_mu.get_mu(xs)
    xs_new = np.arange(len(d)/2, 1.5*len(d))
    
    res  = f_cov.train(th0, xs, d-f_mu.get_mu(xs)) 
    resx = np.squeeze(res.x)
    print "inital    hyperparams : ", th0
    print "optimized hyperparams : ", resx

    f_cov.set_log_hyperparam(resx[0], resx[1], resx[2], resx[3]) ## for periodic
    #f_cov.set_log_hyperparam(resx[0], resx[1:-1], resx[-1]) ## for sq-exp
    f_cov.print_hyperparam()

    gpr = gp.GPR(f_mu, f_cov)
    yo_mu, yo_S = gpr.predict(xs, d, xs_new.copy())
    y_std = np.atleast_2d(np.sqrt(np.diag(yo_S))).T

    fi = np.isfinite(y_std)
    yf = y_std[fi]
    xf = (np.atleast_2d(np.arange(len(y_std))).T)[fi]
    y_std = np.interp(np.arange(len(y_std)), xf, yf)
    y_std = np.atleast_2d(y_std).T

    if plot:
        gp.plot_gpr(get_scaled_x(t,len(yo_mu))+(t[1]-t[0])*xs_new[0],c+yo_mu, y_std, [get_scaled_x(t,len(d)),d+c],
                    xlabel="years", ylabel="CO2 Levels", title="Predicted CO2 Levels (G.P.)")


def predict_sunspots(p=100, plot=True):
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

    if plot:
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

    return d_pred+d_mu, (d+d_mu)[t_pred_idx:]


def train_regressive_nn(d, p=15, epochs=10, alpha=0.5):
    """
    Train a neural-net for regression with:
        p input units
        p/2 hidden units
        1 output unit
    Returns the NN.
    """
    ndat = p*((len(d)-p)/p)-p
    x_train = np.empty((ndat, p))
    for i in xrange(ndat):
        x_train[i,:] = d[i:p+i]
    y_train =  np.atleast_2d(d[p:p+ndat]).T
    assert x_train.shape[0]==y_train.shape[0]

    fname_str = "../data/SS_%d_alpha%0.3f_%s.cp"%(epochs, alpha, 'CE')
    print "Saving results to file : ", fname_str

    start_time = time.time()
    ntrain, n_in = x_train.shape
    arch = [p, p/2, 1] ## neural-net with 1 hidden layer of size p/2.

    print "Training set size = %d"%ntrain

    EPOCHS = epochs
    BATCH_SIZE  = 200
    MIN_GRAD = 1e-4 # if the magnitude of the gradient is smaller than this => stop.
    mag_dW   = 2*MIN_GRAD
    eta    = 1.0
    ETAS   = np.array([eta/(i+1)**alpha for i in xrange(EPOCHS)])
    print "  => learning rates: from %0.3f to %0.3f"%(ETAS[0], ETAS[-1])

    NN = nn.neural_net(arch, [nn.f_tanh(), nn.f_tanh(), nn.f_identity()])

    train_error = []
    obj_error   = []
    run_time    = []

    ## iterate over each epoch:
    for t in xrange(EPOCHS):
      
      if mag_dW < MIN_GRAD:
        print "Stopping at epoch %d due to small gradient = %f"%(t, mag_dW)
        break

      rand_idx = np.arange(ntrain)
      np.random.shuffle(rand_idx)
      n_batch  = int(math.ceil(ntrain / (BATCH_SIZE+0.0))) 
      learn_rate = ETAS[t]

      ## do learning for each mini-batch:
      for i in xrange(n_batch):
        if t!=0:
          sys.stdout.write( "Epoch : % 4d/% 4d | error : train %0.3f | Batch : % 4d/% 4d"%(t+1, EPOCHS, train_error[-1], i+1, n_batch) )
        else:
          sys.stdout.write( "Epoch : %04d/%04d | Batch : %04d/%04d "%(t+1, EPOCHS, i+1, n_batch) )    
        sys.stdout.flush()
        if i!= n_batch:
          idx_batch = rand_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        else:
          idx_batch = rand_idx[i*BATCH_SIZE:]

        x, y = x_train[idx_batch,:], y_train[idx_batch,:]
        mag_dW, _, obj_err = NN.update_batch(nn.squared_error(), x, y, learn_rate)
        mag_dW = np.mean(mag_dW)

        sys.stdout.write('\r')
        sys.stdout.flush()

      if t%10==0:
        err = 0.0
        for i in xrange(x_train.shape[0]):
            err += np.abs(y_train[i,:]-NN.classify(x_train[i,:],regress=True))
        err /= (x_train.shape[0]+0.0)
        train_error.append(err)
        obj_error.append(obj_err)
        run_time.append(time.time()-start_time)

        cp.dump({'train_error':train_error,
                 'obj_error':obj_error,
                 'time': run_time}, open(fname_str,'w'))
    print
    return NN


def predict_sunspots_nn(p, epochs=100, alpha=0.1, plot=True):
    """
    Predict sunspot data using neural-nets (feed-forward).
    Uses last 'p' values as inputs to the NN.
    """
    dfname = osp.join(ddir,"sunspots.mat")
    d = sio.loadmat(dfname)
    t,d = np.squeeze(d['year']), np.squeeze(d['activity'])
    d_mu, d_max = np.mean(d), np.max(d)
    d = (d - d_mu)/d_max ## normalize the data

    id_1990 = int(np.nonzero(t==1990)[0])
    t_train, d_train, t_test, d_test = t[:id_1990], d[:id_1990], t[id_1990:], d[id_1990:]

    nn = train_regressive_nn(d,p=p,epochs=epochs, alpha=alpha)

    def rolling_regression_nn(dd, NN,  n_predict=12):
        """
        Return n_predictions into the future, given
        d entries from the past and the neural-net NN.
        """
        yout = np.empty(n_predict)
        y_prev = dd
        for i in xrange(n_predict):
            yout[i] = NN.classify(y_prev,regress=True).copy()
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
        d_pred[i*12:i*12+n_predict] = rolling_regression_nn(d[t_idx-p:t_idx].copy(), nn, n_predict)

    corr = fftconvolve(d[t_pred_idx:], d_pred[::-1], mode='full')
    i_mid = (len(corr))/2
    i_peak = np.argmax(corr)
    di_step = (i_peak-i_mid)
    print "need to adjust by = %d, p=%d"%(di_step, p)
    di = di_step*(t[1]-t[0])

    d_pred  = d_pred*d_max+d_mu
    d       = d*d_max + d_mu

    if plot:
        plt.plot(corr)
        plt.scatter([i_mid], [corr[i_mid]], c='g', label="mid (ideal peak)")
        plt.scatter([i_peak], [corr[i_peak]], c='r', label="peak")
        plt.title("Cross-Correlation b/w Ground-Truth & Predicted Sunspot Data for Lag Detection")
        plt.legend()
        plt.show()
        
        d_test  = d_test*d_max+d_mu
        d_train = d_train*d_max+d_mu
        
        plt.plot(t_train, d_train, 'b-.', label="ground truth (train)")
        plt.plot(t_test,  d_test, 'r-.', label="ground truth (test)")
        plt.plot(t_pred,  d_pred, 'g', label="prediction")
        plt.title("Sunspot Prediction Using %d-Sample Lagged Feed-Forward Neural Net"%p)
        plt.xlabel("year")
        plt.ylabel("activity")
        plt.legend()
        plt.show()

    return d_pred, d[t_pred_idx:]


def sweep_p_ss(nn=False):
    dfname = osp.join(ddir,"sunspots.mat")
    d = sio.loadmat(dfname)
    t,d = np.squeeze(d['year']), np.squeeze(d['activity'])

    ps= np.array([2,5,10,20,50,100,200,300,500])

    err= np.zeros(len(ps))
    ys = []
    for i in xrange(len(ps)):
        p = ps[i]
        if nn:
            y_p, dgt = predict_sunspots_nn(p, epochs=500, alpha=0.6, plot=False)
        else:
            coeffs = AR_autocorr(d,p=p)
            y_p, dgt = predict_sunspots(p=p,plot=False)
        ys.append(y_p)
        err[i] = rms(y_p, dgt)

    cp.dump([ps,err], open("ss_err_nn.cp" if nn else "ss_err_ar.cp", 'w'))
    plt.plot(ps, np.exp(err), marker='.')
    plt.xlabel('p (auto-regression order)')
    plt.ylabel("rms error")
    if nn:
        ptitle = "Sunspots Neunal Net Error Vs. Model Order"
    else: ptitle = "Sunspots Auto-Regression Error Vs. Model Order"
    plt.title(ptitle)
    plt.show()


if __name__=="__main__":
    choice = sys.argv[1]
    if choice =='0':
        print "Predicting Sunspots using AR(15) model."
        predict_sunspots(p=15, plot=True)
    elif choice =='1':
        print "Plotting sunspots AR(p) models' error function."
        sweep_p_ss(nn=False)
    elif choice == '2':
        print "Prediciting sunspots using neural nets."
        predict_sunspots_nn(15,epochs=20,alpha=0.6,plot=True)
    elif choice =='3':
        print "Plotting sunspots Neural-Net error function. This might take a long time.."
        sweep_p_ss(nn=True)
    elif choice=='4':
        print "Predicting CO2 data using model decomposition."
        predict_CO2(return_full=False, plot=True)
    elif choice=='5':
        print "Prediciting CO2 data using periodic gaussian process regression."
        predict_CO2_gp(plot=True)
    else:
        print "Unkown choice: use a number between 0 and 5. Exiting."
