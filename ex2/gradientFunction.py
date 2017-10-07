import numpy as np

from sigmoid import sigmoid


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    # grad = 1 / m * ((hypothesis - y)' * X)'
    m = len(y)  # number of training examples
    hypothesis = sigmoid(np.dot(X, theta))

    grad = 1 / m * np.array(np.dot(np.array(hypothesis - y), X)).T
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================

    return grad
