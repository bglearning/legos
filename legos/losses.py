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
