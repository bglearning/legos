import numpy as np

from ..losses import CrossEntropyLoss


def test_cross_entropy_loss():
    cross_entropy_loss = CrossEntropyLoss()

    # Case I
    actual = [1.0, 0.0]
    predicted = [1.0, 0.0]
    np.testing.assert_almost_equal(cross_entropy_loss.loss(actual, predicted), 0.0)

    # Case II
    actual = [1.0, 0.0]
    predicted = [0.5, 0.5]
    np.testing.assert_almost_equal(cross_entropy_loss.loss(actual, predicted), - np.log(0.5))

