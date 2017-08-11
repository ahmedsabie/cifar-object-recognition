import numpy as np
import math as m
from copy import deepcopy


class LogisticRegression:
    def __init__(self, theta_init, X, y):
        self.theta = deepcopy(theta_init)
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        # number of training examples, number of features
        self.m, self.n = X.shape

        self.mean = None
        self.std = None

    def get_theta(self):
        return self.theta

    def set_theta(self, new_theta):
        self.theta = deepcopy(new_theta)

    def normalize_features(self, has_intercept_term):
        X = self.X
        if has_intercept_term:
            # remove first column of 1s
            X = np.delete(X, 0, 1)
        # use Z-score normalization
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X -= mean
        X /= std
        if has_intercept_term:
            # add back first column of 1s
            X = np.insert(X, 0, 1, 1)
        self.X = X
        # store Z score values to normalise non training set input
        self.mean = mean
        self.std = std

    def compute_predictions(self):
        return _map(self.X * self.theta, _sigmoid)

    def cost_function(self):
        y = self.y
        predictions = self.compute_predictions()

        y_arr = y.A1
        predictions_arr = predictions.A1

        y_iter = np.nditer(y_arr)
        predictions_iter = np.nditer(predictions_arr)

        # calculate cost iteratively since _log can return -inf
        positives_cost = negatives_cost = 0
        while not y_iter.finished:
            if y_iter[0]:
                positives_cost -= _log(predictions_iter[0])
            else:
                negatives_cost -= _log(1-predictions_iter[0])
            y_iter.iternext()
            predictions_iter.iternext()

        return positives_cost + negatives_cost
    
    def cost_function_gradient(self):
        predictions = self.compute_predictions()
        difference = predictions - self.y
        partial_derivative = difference.transpose() * self.X
        return partial_derivative.transpose()


def _map(matrix, func):
    def map_single(element):
        return func(element)
                             
    vec = np.vectorize(map_single)
    return vec(matrix)


def _sigmoid(x):
    try:
        result = 1.0 / (1 + m.exp(-x))
    except OverflowError:
        # infinity case
        if x > 0:
            result = 1
        # -infinity case
        else:
            result = 0
    return result


def _log(x):
    try:
        result = m.log(x)
    except ValueError:
        # x = 0 case
        result = -1e999
    return result
