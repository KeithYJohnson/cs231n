import numpy as np
from random import shuffle
from ipdb import set_trace as st

# For debugging purposes only.l
naive_margins = np.zeros((500, 10))
first_row_dw = np.zeros((3073, 10))
def svm_loss_naive(W, X, y, reg, first_row_dw=first_row_dw):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    current_example = X[i]
    scores = current_example.dot(W)
    #Print first row score
    score_for_correct_class = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
        naive_margins[i,j] = 0
      margin = scores[j] - score_for_correct_class + 1 # note delta = 1
      naive_margins[i,j] = margin
      if margin > 0:
        # Because we know the margin is greater than 0 we
        # Dont have to worry about max gates in gradient here and can
        # Just use  the (w_j * x_i - w_yi * x_i + someconstant) equaltion
        #
        # del_Li / del_wj =  x_i
        # del_Li / del_yi = -x_i

        dW[:, y[i]] -= current_example
        dW[:, j]    += current_example
        loss += margin
    if i == 0:
        first_row_dw += dW

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train
  dW   += reg * W

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train   = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores_for_correct_classes = scores[np.arange(num_train), y].reshape(num_train,1)

  print(
    'scores_for_correct_classes[0]: ',
     scores_for_correct_classes[0],
     "this should match the earlier print from svm_loss_naive 'score_for_correct_class = scores[y[i]]'"
  )

  margins = np.maximum(0, scores - scores_for_correct_classes + 1)
  #  Because ^^ didnt take into account that we dont want paradoxically
  #  take the diff of the correct class with itself, which is always
  #  zero, plus the constant of 1, resulting in 1 getting added to the
  #  error for each example, we set these to zero on the next line.
  margins[np.arange(num_train), y] = 0

  loss  = np.sum(margins)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
