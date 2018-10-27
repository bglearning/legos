import numpy as np

from legos import activation_functions


def sample_input():
    return np.array([0.0, 0.1, 0.5, 0.9, 1.0])


def test_softmax_2d():
    def softmax_ref(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    inp = sample_input()
    output = softmax_ref(inp)

    input_2d = np.reshape(inp, (-1, len(inp)))
    expected = np.reshape(output, (-1, len(inp)))

    softmax = activation_functions.Softmax()
    np.testing.assert_almost_equal(softmax(input_2d), expected)

    np.testing.assert_almost_equal(softmax.gradient(np.array([1., 2.])),
                                   np.array([[0.19661193, -0.19661193],
                                             [-0.19661193,  0.19661193]]))
