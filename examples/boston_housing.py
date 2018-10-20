import numpy as np

from legos import neural_network
from legos.layers import Linear, Activation
from legos.losses import MeanSquaredError
from legos.optimizers import SGD

from legos.datasets import boston_housing


def main():
    train_x, train_y, test_x, test_y = boston_housing.load_dataset()

    # Normalize Inputs
    train_x_min = np.min(train_x, axis=0)
    train_x_max = np.max(train_x, axis=0)

    train_x = (train_x - train_x_min) / (train_x_max - train_x_min)
    test_x = (test_x - train_x_min) / (train_x_max - train_x_min)

    # Network parameters
    hidden_units = 5
    n_epochs = 500
    learning_rate = 0.001

    nn = neural_network.NeuralNetwork([Linear(n_units=hidden_units, input_size=train_x.shape[1]),
                                       Activation(name='sigmoid'),
                                       Linear(n_units=1, input_size=hidden_units)],
                                      loss=MeanSquaredError(),
                                      optimizer=SGD(lr=learning_rate))
    losses = nn.fit(train_x, train_y, n_epochs=n_epochs)

    print('\nTraining MSE After epoch 1: {:.2f}'.format(losses[0]),
          ', Training MSE After epoch {}: {:.2f}'.format(n_epochs, losses[-1]))

    print('\nTest MSE After epoch {}: {:.2f}'.format(n_epochs, MeanSquaredError().loss(test_y, nn.predict(test_x))))


if __name__ == '__main__':
    main()