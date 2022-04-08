import numpy as np

class RBF:
  '''
  Radial-Basis Function kernel
  '''
  def __init__(self, sigma=1.):
      self.sigma = sigma  ## the variance of the kernel
      self.name = 'RBF'
  def kernel(self,X,Y):
      squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
      return np.exp(-0.5*squared_norm/self.sigma**2)

class linear:
  '''
  Linear kernel
  '''
  def __init__(self,c=0.):
    self.c=c
    self.name='linear'
  def kernel(self,X,Y):
    return X @ Y.T + c

    
class log:
  '''
  Log kernel
  '''
  def __init__(self, poly_degree=1.):
    self.poly_degree = poly_degree
    self.name= 'log'
  def kernel(self,X,Y):
    squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
    squared_norm_d = np.power(squared_norm, self.poly_degree)
    return - np.log(squared_norm_d + 1.)


class poly:
  '''
  Polynomial kernel
  '''
  def __init__(self,alpha=1.,c=0.,d=1.):
    self.alpha=alpha
    self.c=c
    self.d=d
    self.name='poly'
  def kernel(self,X,Y):
    return (self.alpha*(X @ Y.T)+self.c)**self.d

class HIK:
  '''
  Histogram Intersection Kernel (HIK)
  '''
  def __init__(self):
    self.name= 'HIK'
  def kernel(self,X,Y):
    M = np.shape(X)[0]
    N = np.shape(Y)[0]
    G = np.zeros((M,N))
    for i in range(M):
      for j in range(N):
        G[i,j] = np.sum(np.minimum(X[i],Y[j]))
    return G

class polyRBF:
  '''
  Polynomial and RBF kernel
  '''
  def __init__(self,gamma=.5,sigma=1.,alpha=1.,c=0.,d=1.):
    self.gamma=gamma
    self.sigma=sigma
    self.alpha=alpha
    self.c=c
    self.d=d
    self.name='polyRBF'
  def kernel(self,X,Y):
    squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
    RBF = np.exp(-0.5*squared_norm/self.sigma**2)
    poly = (self.alpha*(X @ Y.T)+self.c)**self.d
    return self.gamma*RBF + (1-self.gamma)*poly