"""Boston Housing Dataset.
"""
import numpy as np

from .utils import fetch_file


def load_dataset():

    source_url = ('https://raw.githubusercontent.com/scikit-learn/scikit-learn/master'
                  '/sklearn/datasets/data/boston_house_prices.csv')

    file_path = fetch_file('boston.csv', source_url)

    data = np.genfromtxt(file_path, delimiter=',', dtype=float, skip_header=2)

    # Last of the 14 columns is the target (price)
    X, y = data[:, :-1], data[:, -1:]

    # TrainX, TrainY, TestX, TestY
    return X[:400], y[:400], X[400:], y[400:]

