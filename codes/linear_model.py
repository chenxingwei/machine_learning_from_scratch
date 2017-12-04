```python
#!/usr/bin/env python2.7

"""
This is my codes for linear models, including simple linear regression, multiple linear regression.
"""

class linear_model:
  """
  Implementation linear models
  """
  def __init__(self):
    pass
  
  def train(self, X, y):
    """
    @param X, independent variables. X will np.array with n x m size, n represent number of samples, m represent number of variables.
    example: X = np.array([[1,2],[2,1],[3,3]])
    @param y, dependent variable, np.array, example: y = np.array([1,2,3]) with size n
    len(X) should equal to len(y)
    """
    if len(X) != len(y):
      print "Error, X and y should have the same length."
      return
    # make sure X and y are np.array
    X = np.array(X)
    y = np.array(y)
    # n is number of samples
    n = len(y)
    # below codes to transform X = np.array([1,2,3,4]) to np.array([[1],[2],[3],[4]])
    if len(X.shape) == 1:
      X = X.reshape(n,1)

```
