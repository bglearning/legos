"""Built-in activation functions

The functions are actually classes with a `__call__` method defined on them.
The main purpose is to define the gradient function with the class itself.

Note: The gradient function actually takes in the input!
"""
import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1. - self.__call__(x))


class TanH:
    def __call__(self, x):
        return 2. / (1. + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


ACTIVATIONS = {
    'sigmoid': Sigmoid(),
    'tanh': TanH(),
    'relu': ReLU()
}


def get(activation_identifier):
    """Return activation associated with the identifier.

    Parameters
    ----------
    activation_identifier: str
        String denoting the activation wanted.

    Returns
    -------
    A callable instance of an Activation class with `gradient` implemented as well.
    """
    if activation_identifier is None:
        # Default to ReLU
        return ReLU()
    activation = ACTIVATIONS.get(activation_identifier)
    if activation is None:
        raise ValueError("Couldn't interpret activation identifier: ",
                         activation_identifier)
    return activation

