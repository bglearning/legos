import numpy as np

from legos import activation_functions


class Layer:
    def forward(self, input):
        """Compute the output for the input.
        """
        raise NotImplementedError()

    def backward(self, accum_grad):
        """Compute gradient to propagate backward

        Say, `x` is the layer's input and `y` is its output.
        Now, the layer receives the accumulated gradient dJ/dy i.e
        the gradient with respect to its outputs.

        It needs to return the accumulated gradient dJ/dx i.e the
        gradient with respect to its inputs.

        Also, simultaneously trains the layer if `trainable` is True.

        Parameters
        ----------
        accum_grad: Tensor
            The accumulated gradient with respect to the output of the layer
            Its shape is (batch_size, num_outputs) i.e for each instance, gradient
            with respect to each output.

        Returns
        -------
        new_grad
            The accumulated gradient with respect to the input of the layer
        """
        raise NotImplementedError()


class Linear(Layer):
    """A Linear Layer computes a linear combination of the inputs.

    output = input @ w + b

    Attributes
    ---------
    n_units: int
        Number of neurons in the layer
    input_size: int
        Number of features. The actual input can be of shape (batch_size, input_size).
    trainable: bool
        Whether the layer can be trained, i.e its weights updated, or not.

    """
    def __init__(self, n_units, input_size, trainable=True):
        super().__init__()
        self.input = None
        self.trainable = trainable
        self.optimizer = None
        self.input_size = input_size
        self.n_units = n_units
        self.W = None
        self.b = None

    def initialize(self, optimizer):
        """Initialize Layer.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer to use for the layer

        Returns
        -------
        None
        """
        self.optimizer = optimizer
        self.W = np.random.randn(self.input_size, self.n_units)
        self.b = np.random.randn(self.n_units)

    def forward(self, input):
        # Remember the input for the backward pass
        self.input = input
        return input @ self.W + self.b

    def _backward(self, accum_grad):
        """ Compute the backward pass gradient

        dz/dx = dz/dy * dy/dx
        Here y = Wx + b
        So, dy/dx = W and thus new accum_grad
        dz/dx = dz/dy * W

        Note that we are dealing with tensors here.
        """

        # Remember weight used during forward pass
        W = self.W

        if self.trainable:
            # ----- Gradients with respect to layer params -------

            # For each W_{i} need to sum all x_{bi} * accum_grad{bi}
            grad_W = self.input.T @ accum_grad

            # Need to sum all accum_grad{b}. keepdims to make grad_b broadcast correctly
            grad_b = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.optimizer.update(W, grad_W)
            self.b = self.optimizer.update(self.b, grad_b)

        # Use forward pass Weight W to compute new accum_grad
        return accum_grad @ W.T

    def backward(self, accum_grad):
        return self._backward(accum_grad)


class Activation(Layer):
    """A Layer that applies an activation function to its inputs.

    Attributes
    ---------
    name: str
        Name of the activation function to use.
        Should be available in `activation_functions`.
    """
    def __init__(self, name):
        self.name = name
        self.activation = activation_functions.get(name)
        self.input = None

    def forward(self, input):
        # Remember input for backward pass
        self.input = input
        return self.activation(input)

    def backward(self, accum_grad):
        return accum_grad * self.activation.gradient(self.input)

