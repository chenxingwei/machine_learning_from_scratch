import os, glob,random
import numpy as np

class KMeans:
  """
  K means clustering methods
  """
  def __init__(self):
    """
    @param self.cluster_res, np.array, a list indicating the cluster results.
    @param self.centers, the centers of the K means, np.array with size K x m
    """
    self.cluster_res = None
    self.centers = None
    
  def train(self, X, K, methods="euclidean", n_iter=100):
    """
    Main algorithm for K means cluster.
    @param X, input data, np.array, example np.array([[1,2,3,..],[2,3,4,...],...]) with n x m size, n samples, each sample have m features.
    @param K, the hyper-parameter for k means clustering.
    @param methods, methods to calculate the distance between points
    @param n_iter, number of iteration to run
    @return, fill the self.cluster_res
    """
    all_methods = set(["euclidean"])
    if methods not in all_methods:
      print "Error, the methods much be one of ", all_methods
      return
    n = len(X)
    m = len(X[0])
    indexes = range(n)
    samples = random.sample(indexes, K)
    self.centers = X[samples]
    if methods == "euclidean":
      distance = np.sqrt(np.sum(np.square(X - self.centers.rekshape(K, 1, m)), axis=2))
      self.cluster_res = distance.argmin(axis=0)
      for i in range(n_iter):
        for j in range(K):
          tmparray = X[self.cluster_res==j]
          self.centers[j] = np.mean(tmparray,axis=0)
        distance = np.sqrt(np.sum(np.square(X - self.centers.rekshape(K, 1, m)), axis=2))
        self.cluster_res = distance.argmin(axis=0)
