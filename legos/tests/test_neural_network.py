import numpy as np

from legos import neural_network
from legos.layers import Linear
from legos.losses import MeanSquaredError
from legos.optimizers import SGD


def test_linear_regression():
    """Test for a Linear Regression i.e
    a Neural Network with a single Linear Layer.
    """

    X = np.array([[1., 2.],
                  [3., 4.],
                  [1., 1.],
                  [5., 4.],
                  [2., 3.],
                  [-1.5, 2.1],
                  [0.5, -0.3]])

    real_w = np.array([[1.], [1.]])
    real_b = np.array([[0.5]])

    y = X @ real_w + real_b

    nn = neural_network.NeuralNetwork([Linear(n_units=1, input_size=X.shape[1])],
                                      loss=MeanSquaredError(),
                                      optimizer=SGD(lr=0.01))
    errors = nn.fit(X, y, n_epochs=1000)

    w = nn.layers[0].W
    b = nn.layers[0].b

    # Shape of the parameters
    assert w.shape == (2, 1)
    assert b.shape == (1, 1)

    # Value of the parameters
    np.testing.assert_array_almost_equal(w, real_w)
    np.testing.assert_array_almost_equal(b, real_b)

    # Final training error should be zero
    np.testing.assert_almost_equal(errors[-1], 0.0)

