class KNearestNeighbor:
  """
  Implementation the most straight forward KNN algorithms
  """
  def __init__(self):
    self.trainX = None
    self.trainY = None
    
  def train(self, X, y):
    """
    Just store the training data
    @param X, n times m size with n samples and each sample have m paramters, for example: X = np.array([[1,2],[3,4],[2,3],...])
    @param y, list with size n, for example: y = np.array([1,2,3,4,...])
    """
    self.trainX = X
    self.trainY = y
    self.predY = None
    
  def test(self, X, K=3, dist_type="euclidean"):
    """
    @param X, n times m size with n samples and each sample have m paramters, for example: X = np.array([[1,2],[3,4],[2,3],...])
    @param K, hyper parameter for KNN algorithm
    @param dist_type, the type to calculate distance
    @return, Ypred, the predicted categories for the n samples
    """
    distance_types = set(["euclidean", "abs"])
    if dist_type not in distance_types:
      print "Error, the distance types should only be", distance_types
      return
    X = np.array(X)
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype = self.trainY.dtype)
    
    for i in xrange(num_test):
      if dist_type == "euclidean":
        distances = np.sum(np.abs(self.trainX - X[i, :]), axis = 1)
      else:
        # did not sqrt because it's the same
        distances = np.sum(np.square(self.trainX - X[i, :]), axis = 1)
      sort_index = np.argsort(distances)
      min_index = sort_index[:K]
      k_ytr = [self.trainY[tmp_index] for tmp_index in min_index]
      pred, pred_count = mode(k_ytr)
      if pred_count[0] == 1:
        Ypred[i] = -1
      else:
        Ypred[i] = pred[0]
    self.predY = Ypred
    return Ypred
