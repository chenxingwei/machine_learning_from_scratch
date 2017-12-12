import os, glob, math
import numpy as np
import random

class linear_regression_GDL2:
  """
  linear regression with OLS loss function plus L2-norm regularization, and solving by gradient descent methods.
  """
  def __init__(self):
    self.rmse = None
    pass
  
  def train(self, X, y, learning_rate=0.001, epoch=30000, tol=1e-4, methods="OLS", C=0.1):
    """
    If data not the required format, this function not works
    @param X, independent variable, example: X = np.array([[1,2,],[1,3,],[1,4], ...])
    @param y, dependent variable, example y = np.array([[1],[2],[3],[4],...])
    Both X and y should be an array.
    @param learning_rate, step size for gradient descent
    @param epoch, times the training process go over the whole datasets
    @param tol, the stop critera, when loss function did not optimize more than tol for 10 times, stop training
    @param methods, the loss function, OLS stands for ordinary least square loss function
    @return, this function will return weight vector w and bias parameter b
    """
    n = len(X)
    self.n = n
    m = len(X[0])
    if len(y) != n:
      print "Error, the input X, y should be the same length, while you have len(X)=%d and len(y)=%d"%(n, len(y))
    
    # following codes reformat the input
    X = np.array(X)
    y = np.array(y)
    
    if len(y.shape) != 1:
      y = y.reshape(n)
    
    # add a 1 column for the bias variable
    X = np.column_stack((X, np.ones(n)))
    self.X = X
    self.y = y
    
    # get the parameter w and b, first initize
    theta = np.random.randn(m+1)
    num = 0
    loss1 = self.getLoss(theta)
    for i in xrange(epoch):
      theta -= learning_rate * self.getGradient(theta, C)
      loss2 = self.getLoss(theta, methods)
      if abs(loss2-loss1) < tol:
        num += 1
      else:
        num = 0
      loss1 = loss2
      self.epoch = i
      #if num >= 10:
      #  break
    self.w = theta[:-1]
    self.b = theta[-1]
    return theta
    
  def getLoss(self, theta, methods="OLS"):
    """
    @param theta, the parameters for the linear model
    @return the loss of this model with given parameters
    """
    if methods == "OLS":
      # divide self.n first to avoid float overflow to some extent.
      return np.sum(np.square((self.X.dot(theta) - self.y)/self.n) * self.n)
    else:
      print 'Error, methods now can only be "OLS"'
  
  def getGradient(self, theta, C):
    """
    Get the gradient of the loss function using parameter theta
    """
    delta = np.sum(self.X.T * (self.X.dot(theta)-self.y), axis=1) + C * theta
    return delta
  
  def predict(self, X, y=None):
    """
    @param X, independent variable.
    Example: X = np.array([[1,2,],[1,3,],[1,4], ...])
    @param y, dependent variable, example y = np.array([[1],[2],[3],[4],...])
    @return, return the predicted dependent variables for input X, or if y is specified, also calculate the RMSE (root mean square error)
    """
    n = len(X)
    X = np.array(X)
    y_pred = X.dot(self.w) + self.b
    self.y_pred = y_pred
    if y != None:
      if n != len(y):
        print "Error, the input X, y should be the same length, while you have len(X)=%d and len(y)=%d"%(n, len(y))
      y = np.array(y)
      y_pred = y_pred.reshape(n)
      if len(y.shape) != 1:
        y = y.reshape(n)
      self.rmse = np.sqrt(np.sum(np.square(y-y_pred))) / n
    return self.y_pred
