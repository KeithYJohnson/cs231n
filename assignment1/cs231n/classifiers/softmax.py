import numpy as np
from random import shuffle
from ipdb import set_trace as st

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  num_train = X.shape[0]

  # for i in range(num_train):
  #     row = scores[i, :]
  #     row_class = y[i]
  #     row_class_score = row[row_class]
  #
  #     num = np.exp(row_class_score)
  #     denom = np.sum(np.exp(row))
  #
  #     probability_of_correct_class = num / denom
  #     loss = -np.log(probability_of_correct_class)

  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in xrange(num_train):
    scores = X[i:].dot(W)
    scores -= np.max(scores)
    correct_class = y[i]
    normalize_scores = np.exp(scores) / np.sum(np.exp(scores))
    st()
    loss += - np.log(normalize_scores[y[correct_class]])


  loss /= num_train

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
