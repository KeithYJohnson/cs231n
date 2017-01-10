import pytest
import numpy as np
from pdb import set_trace as st
from linear_svm import svm_loss_naive, svm_loss_vectorized

x = np.array([
    [1,0,2],
    [1,2,0],
    [1,0,3]
])

w = np.array([
    [2,1,3],
    [1,1,2],
    [2,0,1]
])

y=[0,1,2]
scores = x.dot(w)
num_train = x.shape[0]
# I hand calc'd the loss summation to be 10.


expected_loss = 10.0 / num_train
def test_simple_loss():
    actual_loss, grad  = svm_loss_naive(w, x, y, 0)
    assert(actual_loss == expected_loss)


reg_strength = 0.01
reg_penalty = 0.5 * reg_strength * np.sum(w * w)
expected_loss_w_reg = expected_loss + reg_penalty
def test_simple_loss_with_regularization():
    actual_loss_w_reg, grad  = svm_loss_naive(w, x, y, reg_strength)
    assert(actual_loss_w_reg == expected_loss_w_reg)

def test_simple_vectorized_loss():
    actual_loss, grad  = svm_loss_vectorized(w, x, y, 0)
    assert(actual_loss == expected_loss)
