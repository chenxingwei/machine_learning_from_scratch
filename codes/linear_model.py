import os, glob, math
import numpy as np

class linear_model:
  """
  linear model class, this is linear models, such as linear regression, logistic regression
  """
  def __init__(self):
    self.rmse = None
    pass
  
  def train(self, X, y):
    """
    If data not the required format, this function not works
    @param X, independent variable, example: X = np.array([[1,2,],[1,3,],[1,4], ...])
    @param y, dependent variable, example y = np.array([[1],[2],[3],[4],...])
    Both X and y should be an array.
    @return, this function will return weight vector w and bias parameter b
    """
    n = len(X)
    m = len(X[0])
    if len(y) != n:
      print "Error, the input X, y should be the same length, while you have len(X)=%d and len(y)=%d"%(n, len(y))
    
    # following codes reformat the input, this function only for simple linear regression
    X = np.array(X)
    y = np.array(y)
    
    if len(y.shape) != 2:
      y = y.reshape(n,1)
    
    # add a 1 column for the bias variable
    X = np.column_stack((X, np.ones(n)))
    
    # get the parameter w and b
    theta=np.array(np.mat(X.T.dot(X)).I.dot(X.T).dot(y))
    self.w = theta[:-1]
    self.b = theta[-1]
    return theta
  
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
  
  
