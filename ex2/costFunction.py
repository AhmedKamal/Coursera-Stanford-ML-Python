import numpy as np

from sigmoid import sigmoid


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # Initialize some useful values
    m = y.size  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    # =============================================================
    #

    Z = np.dot(X, theta)

    hypothesis = sigmoid(Z)
    part1 = np.dot((y - 1), np.log(1 - hypothesis))
    part2 = np.dot(y, np.log(hypothesis))
    J = (1 / m) * (part1 - part2)

    return J
