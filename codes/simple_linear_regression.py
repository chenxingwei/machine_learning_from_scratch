import os, glob, math
import numpy as np

class simple_linear_regression:
  """
  linear model class, this is only for simple linear regression
  """
  def __init__(self):
    self.rmse = None
    pass
  
  def train(self, X, y):
    """
    @param X, independent variable, here requires the X to have only one variable.
    Example: X = np.array([1,2,3,4,...])
    @param y, dependent variable, example y = np.array([1,2,3,4,...])
    @return, this function will return weight paramether w and bias parameter b
    """
    n = len(X)
    if len(y) != n:
      print "Error, the input X, y should be the same length, while you have len(X)=%d and len(y)=%d"%(n, len(y))
    
    # following codes reformat the input, this function only for simple linear regression
    X = np.array(X)
    y = np.array(y)
    if len(X.shape) > 1:
      X = X.reshape(n)
    if len(y.shape) > 1:
      y = y.reshape(n)
    
    # get the parameter w and b
    sumx = np.sum(X)
    sumy = np.sum(y)
    xy = X.dot(y)
    xsquare = X.dot(X)
    
    w = (n*xy - sumx*sumy) / (n*xsquare - sumx**2)
    b = (sumy - w * sumx) / n
    self.w = w
    self.b = b
    return w, b
  
  def predict(self, X, y=None):
    """
    @param X, independent variable, here requires the X to have only one variable.
    Example: X = np.array([1,2,3,4,...])
    @param y, dependent variable, example y = np.array([1,2,3,4,...])
    @return, return the predicted dependent variables for input X, or if y is specified, also calculate the RMSE (root mean square error)
    """
    n = len(X)
    X = np.array(X)
    if len(X.shape) > 1:
      X = X.reshape(n)
    y_pred = self.w * X + self.b
    self.y_pred = y_pred
    if y != None:
      if n != len(y):
        print "Error, the input X, y should be the same length, while you have len(X)=%d and len(y)=%d"%(n, len(y))
      y = np.array(y)
      if len(y.shape) > 1:
        y = y.reshape(n)
      self.rmse = np.sqrt(np.sum(np.square(y-y_pred))) / n
    return y_pred

