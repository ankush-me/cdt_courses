import numpy as np
import matplotlib.pyplot as plt
from nn import *


def get_data(Nlag=250, mean_window=50, ntrain=10000, ntest=2000):
  """
  Load the data from the mat file,
  pre-process it (label and normalize), split into test and training.
  """
  dat  = sio.loadmat('../data/finPredProb.mat')
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


def predict_finance(epochs, alpha, f_err=cross_entropy_error):
  """
  Train a neural-net for financial prediction.
    epochs : number of iterations of back-propagation
    alpha  : learning rate
    f_err  : error-rate metric
  """
  x_train, y_train, x_test, y_test = get_data(Nlag=250, mean_window=50, ntrain=10000, ntest=2000)
  label_train, label_test = np.argmax(y_train,axis=1), np.argmax(y_test,axis=1)

  fname_str = "../data/FinanceDeep2_%d_alpha%0.3f_%s.cp"%(epochs, alpha, 'MSE' if f_err==squared_error else 'CE')
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

def plot_finance_nn():
  fnames = ['../data/FinanceDeep2_1000_alpha0.500_CE.cp']
  d_ce = cp.load(open(fnames[0],'r'))
  M = len(d_ce['test_error'])

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
    predict_finance(epochs=10, alpha=0.5, f_err=cross_entropy_error)
  elif sys.argv[1]=='1':
    plot_finance_nn()
