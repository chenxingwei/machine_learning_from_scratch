import os, glob, math
import numpy as np
import random

class logistic_regression_GD:
  """
  logistic regression with crossentropy loss function, and solving by gradient descent methods.
  """
  def __init__(self):
    self.w = None
    self.b = None
    self.y_pred = None
  
  def train(self, X, y, learning_rate=0.001, epoch=30000, tol=1e-4, methods="crossentropy"):
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
      print "Error, the input X, y should be the same length, \
      while you have len(X)=%d and len(y)=%d"%(n, len(y))
    
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
    for i in xrange(epoch):
      theta -= learning_rate * self.getGradient(theta, methods)
      self.epoch = i
    self.w = theta[:-1]
    self.b = theta[-1]
    return theta
  
  def getGradient(self, theta, methods="crossentropy"):
    """
    Get the gradient of the loss function using parameter theta
    """
    methods_from = ["crossentropy"]
    if methods == "crossentropy":
      a = 1.0 / (1+np.exp(-1.0*(self.X.dot(theta))))- self.y
      a = a.reshape(self.n,1)
      delta = np.sum(self.X*a, axis=0)
    else:
      print "Error, the methods parameter can only be ", methods_from
    return delta
  
  def predict(self, X, y=None):
    """
    @param X, independent variable.
    Example: X = np.array([[1,2,],[1,3,],[1,4], ...])
    @param y, dependent variable, example y = np.array([[1],[2],[3],[4],...])
    @return, return the predicted dependent variables for input X, or if y is specified
    """
    n = len(X)
    X = np.array(X)
    y_pred = X.dot(self.w) + self.b
    self.y_pred = y_pred
    if y != None:
      if n != len(y):
        print "Error, the input X, y should be the same length, \
        while you have len(X)=%d and len(y)=%d"%(n, len(y))
      y = np.array(y)
      y_pred = y_pred.reshape(n)
      if len(y.shape) != 1:
        y = y.reshape(n)
      self.rmse = np.sqrt(np.sum(np.square(y-y_pred))) / n
    return self.y_pred
