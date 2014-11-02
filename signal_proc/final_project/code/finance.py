import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cPickle as cp
import os.path as osp
import math, sys, time

"""
Financial Data prediction using neural-nets.
"""

eps = np.spacing(0)

def error_rate(y_pred, y_gt):
  y_pred = y_pred.flatten()
  y_gt   = y_gt.flatten()
  return  np.sum(y_pred!=y_gt)/(len(y_gt)+0.0)


class squared_error:
  """
  err = 0.5*||y-y_pred||^2
  """
  def val(self, y,y_pred):
    return 0.5*np.linalg.norm(y-y_pred)

  def deriv(self, y, y_pred):
    return (y_pred-y)

class cross_entropy_error:
  """
  - sum_i yi*log(yi_pred) + (1-yi)*log(1-yi_pred) 
  """
  def val(self, y,y_pred):
    y_pred += eps
    return -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
  
  def deriv(self, y, y_pred):
    return (y_pred-y)/((y_pred*(1.0-y_pred)) + eps)

class f_sigmoid:
  """
  The sigmoid function.
  """
  def val(self, x):
    return 1./(1 + np.exp(-x))
  
  def deriv(self, x):
    assert x.ndim==1
    yx = self.val(x)
    return yx*(1-yx)

class f_tanh:
  """
  The tanh function.
  """
  def val(self, x):
    return np.tanh(x)

  def deriv(self, x):
    assert x.ndim==1
    yx = self.val(x)
    return 1-yx**2


class neural_net:
  """
  Class to represent fully-connected NN.
  -- Uses back-propagation for learning.
  """

  def __rand_high_low__(self, high, low, *shape):
    m = np.random.rand(*shape)
    return low + (high-low)*m

  def __init__(self, arch, f_nls):
    """
    arch : A list of the number of neurons in each layer.
           For example : Arch=[100,150,10] means, that the
           NN has 100 units in the input layer, 150 in the
           hidden layer and 10 in the output layer.
           
           The list can be of arbitrary length.
    
    f_nl : A list of non-linear functions to use per layer.
           If NONE, the SIGMOID function is used.

    err : String in { "ms" : for mean-squared error ||y-y_pred||^2
                      "ce" : for cross-entropy error sum_{out-units} :
                              y_i*log(y_i_pred) + (1-y_i)*log(1-y_i_pred) }
    """
    assert len(arch) > 1
    self.nlayers = len(arch)-1
    self.arch = arch
    self.weight = []
    self.bias   = []

    if f_nls:
      self.f_nls = f_nls
    else:
      self.f_nls = self.nlayers*[f_sigmoid()]

    ## initialize random weights and biases:
    for l in xrange(self.nlayers):
      n_in  = self.arch[l]
      n_out = self.arch[l+1]
      rand_lim = 1.0/(n_in+1.0)

      W = self.__rand_high_low__(-rand_lim, rand_lim, n_out,n_in)
      self.weight.append(W)

      b = self.__rand_high_low__(-rand_lim, rand_lim, n_out)
      self.bias.append(b)

  def layer_forward(self, i, xin):
    """
    returns the :
      1. Activation = Wx + b
      2. output = f(Wx+b)
    of the i_th layer with x_in as the input. 
    """
    assert 0<=i< self.nlayers, "cannot do forward : layer index out of range."
    W = self.weight[i]
    b = self.bias[i]
    S = W.dot(xin) + b
    return S, self.f_nls[i].val(S)

  def forward(self, xin):
    """
    returns end-to-end activation (S) output (f(S))
    at each layer given the input xin.
    """
    S, fS = self.layer_forward(0, xin)
    S, fS = [S], [fS]
    for i in xrange(1, self.nlayers):
      Si, fSi = self.layer_forward(i, fS[-1])
      S.append(Si)
      fS.append(fSi)
    return S, fS

  def classify(self, xin):
    """
    Returns the armgax of the output units.
    """
    _, fS = self.layer_forward(0, xin)
    for i in xrange(1, self.nlayers):
      _, fS = self.layer_forward(i, fS)    
    return np.argmax(fS)

  def layer_backward(self, i, d_back, S_i):
    """
    Computes the delta for layer i.

    S_i    : Activation of the current layer = Wx + b
    d_back : delta for the layer (i+1).

    Note: DELTA for the last layer is special.
    """
    assert i!=self.nlayers-1, "No Delta for the last layer."

    W = self.weight[i+1]
    f_prime = self.f_nls[i].deriv(S_i)
    return (W.T.dot(d_back))*f_prime

  def backward(self, f_err, y_gt, S, fS):
    """
    f_err : function to use for error-calculation at the output layer
    y_gt  : The ground truth output (a vector of length = number of output units).
    S     : the forward activations = Wx+b of each layer
    fS    : the output of each layer = f(Wx+b)
    """
    deltas = []

    ## compute the delta for the last layer:
    ilast  = self.nlayers-1
    d_last = f_err.deriv(y_gt, fS[ilast]) * self.f_nls[ilast].deriv(S[ilast])
    deltas.append(d_last)

    ## propagate back:
    for l in xrange(self.nlayers-2, -1, -1):
      deltas.append(self.layer_backward(l, deltas[-1], S[l]))

    #deltas.reverse()
    return deltas, f_err.val(y_gt, fS[ilast])

  def update_batch(self, f_err, x_in, y_in, alpha=0.1):
    """
    Update the weights using batch-gradient descent.
  
    The total batch-error is defined as :
    
    sum_{training_set} f_err(y_gt, y_pred)
    ---------------------------------------
       # elements in the training set

    f_err: the error function to use for training.
    x_in : a matrix of size (S x n_in) :
             S is the size of the batch,
             n_in is the number of input units.
    y_in : a matrix of size (S x n_out):
             S is the size of the batch,  
             n_out is the number of output units.
    alpha: the learning rate: W = W - alpha*dW
                              b = b - alpha*b
    """
    B,n_in  = x_in.shape
    B,n_out = y_in.shape

    dW = [np.zeros(W.shape) for W in self.weight] ## change in the weights for each layer.
    db = [np.zeros(b.shape) for b in self.bias]   ## change in the bias for each layer.

    for si in xrange(B): ## iterate over each example
      x,y = x_in[si,:], y_in[si,:]
      S, fS = self.forward(x)
      delta, obj_err = self.backward(f_err, y, S, fS)

      ## compute the deviations:
      for l in xrange(self.nlayers):
        dW[l] += np.outer(delta[len(delta)-1-l], x if l==0 else fS[l-1])
        db[l] += delta[len(delta)-1-l]

    ## normalize by the size of the batch:
    mag_dW, mag_db = [],[]
    for l in xrange(self.nlayers):
      self.weight[l] -= alpha/B*dW[l]
      self.bias[l]   -= alpha/B*db[l]
      mag_dW.append(np.linalg.norm(dW[l]))
      mag_db.append(np.linalg.norm(db[l]))

    return mag_dW, mag_db, obj_err


def get_data(Nlag=250, mean_window=50, ntrain=10000, ntest=2000):
  """
  Load the data from the mat file,
  pre-process it (label and normalize), split into test and training.
  """
  dat  = sio.loadmat('finPredProb.mat')
  dat  = np.squeeze(dat['ttr'])
  N    = len(dat)
  mu   = np.mean(dat)
  mdat = dat - mu

  ddat  = mdat[1:]-mdat[:-1]
  nddat = ddat/(4.999e-5)
  nddat_i = nddat.astype(int)
  nddat = nddat/np.abs(np.max(nddat))

  rand_idx = np.random.choice(N-(Nlag+1),ntrain+ntest,replace=False)+Nlag
  ntrain_idx, ntest_idx = rand_idx[:ntrain], rand_idx[ntrain:]

  x_train = np.empty((ntrain, Nlag+Nlag/mean_window))
  y_train = np.zeros((ntrain, 2*4+1))
  for i in xrange(ntrain):
    dat_idx = ntrain_idx[i]
    x_dat  = nddat[dat_idx-Nlag:dat_idx]
    mu_dat = mdat[dat_idx-Nlag:dat_idx]
    mu_windowed = np.mean(mu_dat.reshape(Nlag/mean_window, mean_window), axis=1)
    x_train[i,:] = np.r_[x_dat, mu_windowed]
    y_train[i, nddat_i[dat_idx]+4]   = 1.0


  x_test = np.empty((ntest, Nlag+Nlag/mean_window))
  y_test = np.zeros((ntest, 2*4+1))
  for i in xrange(ntest):
    dat_idx = ntest_idx[i]
    x_dat  = nddat[dat_idx-Nlag:dat_idx]
    mu_dat = mdat[dat_idx-Nlag:dat_idx]
    mu_windowed = np.mean(mu_dat.reshape(Nlag/mean_window, mean_window), axis=1)
    x_test[i,:] = np.r_[x_dat, mu_windowed]
    y_test[i, nddat_i[dat_idx]+4]   = 1.0

  return x_train, y_train, x_test, y_test


def get_NN_error_rate(NN, Xin, Ygt):
  """
  Returns the error rate, given a neural-net NN,
  a matrix of inputs Xin ( Nxn_in )
  and the ground-truth labels Ygt (size N).
  """
  N, n_in = Xin.shape
  y_pred = np.empty(Ygt.size)
  for i in xrange(N):
    x = Xin[i,:]
    y_pred[i] = NN.classify(x)
  return error_rate(y_pred, Ygt)


def predict_finance(epochs, alpha, f_err=cross_entropy_error):
  """
  Train a neural-net for financial prediction.
    epochs : number of iterations of back-propagation
    alpha  : learning rate
    f_err  : error-rate metric
  """
  x_train, y_train, x_test, y_test = get_data(Nlag=250, mean_window=50, ntrain=10000, ntest=2000)
  label_train, label_test = np.argmax(y_train,axis=1), np.argmax(y_test,axis=1)

  fname_str = "FinanceDeep2_%d_alpha%0.3f_%s.cp"%(epochs, alpha, 'MSE' if f_err==squared_error else 'CE')
  print "Saving results to file : ", fname_str

  start_time = time.time()
  ntrain, n_in = x_train.shape
  ntest, _    = x_test.shape
  arch = [n_in, 100, 50, 10, y_train.shape[1]] ## neural-net with 2 hidden layers of size 100 and 50.


  print "Training set size = %d"%ntrain

  EPOCHS = epochs
  BATCH_SIZE  = 20
  MIN_GRAD = 1e-4 # if the magnitude of the gradient is smaller than this => stop.
  mag_dW   = 2*MIN_GRAD
  eta    = 1.0
  ETAS   = np.array([eta/(i+1)**alpha for i in xrange(EPOCHS)])
  print "  => learning rates: from %0.3f to %0.3f"%(ETAS[0], ETAS[-1])

  NN = neural_net(arch, [f_tanh(), f_tanh(),  f_tanh(), f_sigmoid()])

  train_error = []
  test_error  = []
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
        sys.stdout.write( "Epoch : % 4d/% 4d | error : train %0.3f test %0.3f | Batch : % 4d/% 4d"%(t+1, EPOCHS, train_error[-1], test_error[-1], i+1, n_batch) )
      else:
        sys.stdout.write( "Epoch : %04d/%04d | Batch : %04d/%04d "%(t+1, EPOCHS, i+1, n_batch) )    
      sys.stdout.flush()
      if i!= n_batch:
        idx_batch = rand_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
      else:
        idx_batch = rand_idx[i*BATCH_SIZE:]

      x, y = x_train[idx_batch,:], y_train[idx_batch,:]
      mag_dW, _, obj_err = NN.update_batch(f_err(), x, y, learn_rate)
      mag_dW = np.mean(mag_dW)

      sys.stdout.write('\r')
      sys.stdout.flush()

    if t%10==0:
      train_error.append(get_NN_error_rate(NN, x_train, label_train))
      test_error.append(get_NN_error_rate(NN, x_test, label_test))
      obj_error.append(obj_err)
      run_time.append(time.time()-start_time)

      cp.dump({'test_error':test_error,
               'train_error':train_error,
               'obj_error':obj_error,
               'time': run_time}, open(fname_str,'w'))

def plot():
  fnames = ['FinanceDeep2_1000_alpha0.500_CE.cp']
  
  d_ce = cp.load(open(fnames[0],'r'))
  
  plt.hold(True)

  #N = len(d_mse['test_error'])
  M = len(d_ce['test_error'])

  #plt.plot(10*np.arange(N), d_mse['test_error'],  '-o'  , color='k',   label='MSE test')
  #plt.plot(10*np.arange(N), d_mse['train_error'], '-s'  , color='0.6', label='MSE train')
  #plt.plot(10*np.arange(N), d_mse['obj_error'],  '-' , color='r', label='MSE Loss')
  plt.plot(10*np.arange(M), d_ce['test_error']  , '-^' , color='k',   label='CE test')
  plt.plot(10*np.arange(M), d_ce['train_error'],  '-*' , color='0.6', label='CE train')
  plt.plot(10*np.arange(M), d_ce['obj_error'],  '-' , color='b', label='CE Loss')
  
  plt.xlabel('num epoch')
  plt.ylabel('error rate')
  plt.ylim([0,1.0])
  plt.legend()

  plt.title('2-Hidden Layers Misclassification Rate')
  plt.show()


if __name__=='__main__':
  if sys.argv[1]=='0':
    predict_finance(epochs=1000, alpha=0.5, f_err=cross_entropy_error)
  elif sys.argv[1]=='1':
    plot()
