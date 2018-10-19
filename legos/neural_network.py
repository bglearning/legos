class NeuralNetwork:
    """Neural Network.

    Attributes
    ---------
    layers: list of Layer
        Layers that make up the network.
    optimizer: Optimizer object
        Optimizer to use.
    loss: Loss object
        Loss to use
    """
    def __init__(self, layers, optimizer, loss):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def _forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def _backward_pass(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _initialize(self):
        """Initialize all layers if they can be/need to be initialized.
        """
        for layer in self.layers:
            if hasattr(layer, "initialize"):
                layer.initialize(self.optimizer)

    def fit(self, X, y, n_epochs, initialize=True):
        """Fit the parameters of the model.

        Parameters
        ----------
        X: Tensor
            The inputs
        y: Tensor
            The expected outputs
        n_epochs: int
            The number of epochs to train.
        initialize: bool
            Whether to initialize the layers or not.

        Returns
        -------
            A list of training errors for each epoch.
        """
        if initialize:
            self._initialize()

        training_loss = []
        for e in range(n_epochs):
            predicted = self._forward_pass(X)

            loss = self.loss.loss(y, predicted)
            training_loss.append(loss)

            grad = self.loss.grad(y, predicted)
            self._backward_pass(grad)

        return training_loss
