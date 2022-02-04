import numpy as np
import scipy.optimize as sopt
from utilities import slog, valign


def sigmoid(x):
    "Numerically stable sigmoid function."

    def _positive_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x):
        exp = np.exp(x)
        return exp / (exp + 1)

    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


def logreg(X, Theta):
    return sigmoid(X @ Theta)


def logreg_loss(Theta, X, Y, alpha):
    "Logistic regression cost suitable for use with fmin_l_bfgs."

    # Reshape Theta into a column vector - lBFGS gives us a flat array
    ThetaR = valign(Theta)

    hx = logreg(X, ThetaR)
    nll = -np.sum(Y.T @ slog(hx) + (1-Y).T @ slog(1-hx)) + alpha * (ThetaR @ ThetaR.T)
    grad = X.T @ (hx - Y)

    # Reshape grad into the shape of Theta, for fmin_l_bfsgb to work
    return nll, grad.reshape(Theta.shape)


class LogisticRegression:
    def __init__(self, X, Y, alpha=0, theta: np.ndarray = None):
        if theta is not None:
            self.theta = theta
            return

        assert X.shape[0] == Y.shape[0], "shape mismatch!"
        assert Y.shape[1] == 1, "expected a vertical vector for Y!"

        # Call a solver
        self.theta = sopt.fmin_l_bfgs_b(
            lambda Theta: logreg_loss(Theta, X, Y, alpha), np.zeros(X.shape[1])
        )[0]

    def error(self, X, Y):
        "Return percentage of failed predictions."
        assert X.shape[0] == Y.shape[0], "shape mismatch!"
        assert Y.shape[1] == 1, "expected a vertical vector for Y!"

        preds = valign(np.round(self.__call__(X)).astype(int))
        return np.sum(np.abs(Y - preds)) / len(Y)

    def __call__(self, X):
        assert X.shape[1] == self.theta.shape[0], "theta shape mismatch!"
        return logreg(X, self.theta)

    def __repr__(self):
        return str(self.theta)
