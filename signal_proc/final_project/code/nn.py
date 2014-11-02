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

