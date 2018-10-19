class Optimizer:
    """Updates the given parameter.

    Basically, trains the network.
    """
    def update(self, w, grad_wrt_w):
        """Update
        Parameters
        ----------
        w: Tensor
            The current value of the weights
        grad_wrt_w: Tensor
            The gradient of the Loss with respect to the weights.

        Returns
        -------
        A Tensor that's the new updated value of the weights.
        """
        raise NotImplementedError()


class SGD(Optimizer):
    """Stochastic Gradient Descent

    It could actually be just called Gradient Descent as
    it doesn't really care if the weights supplied are
    for batches or single instances.

    Attributes
    ---------
    lr: float
        The learning rate
    """
    def __init__(self, lr=0.01):
        self.learning_rate = lr

    def update(self, w, grad_wrt_w):
        return w - self.learning_rate * grad_wrt_w
