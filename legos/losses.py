"""
Loss Functions
"""
import numpy as np


class Loss:
    def loss(self, actual, predicted):
        raise NotImplementedError()

    def grad(self, actual, predicted):
        raise NotImplementedError()


class MeanSquaredError(Loss):
    """Mean Squared Error
    """
    def loss(self, actual, predicted):
        return np.mean((predicted - actual) ** 2) / 2

    def grad(self, actual, predicted):
        return predicted - actual


class CrossEntropyLoss(Loss):
    """Categorical Cross Entropy Loss

    Parameters
    ----------
    epsilon: float, optional (default=1e-15)
        The value to use for clipping the probabilities.
    """
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def _clip(self, values):
        return np.clip(values, self.epsilon, 1 - self.epsilon)

    def loss(self, actual, predicted):
        predicted = self._clip(predicted)
        return - np.sum(actual * np.log(predicted), axis=-1)

    def grad(self, actual, predicted):
        predicted = self._clip(predicted)
        return - np.multiply((1 / predicted), actual)
