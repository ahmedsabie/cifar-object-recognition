import numpy as np
import math as m
class LogisticRegression:
    
    def __init__(self, theta_init, X, y):
        self.theta = theta_init
        self.X = X
        self.y = y
        # number of training examples
        self.m = len(X)
        # number of features
        self.n = len(X[0])

    def compute_predictions(self):
        return _map(self.X * self.theta, _sigmoid)

    def cost_function(self):
        y = self.y
        predictions = self.compute_predictions()
        positives_cost = -np.multiply(
            y, _map(predictions, m.log)).sum()
        negatives_cost = -np.multiply(
            1-y, _map(predictions, lambda x: m.log(1-x))).sum()
        return positives_cost + negatives_cost
    
    def cost_function_gradient(self):
        predictions = self.compute_predictions()
        difference = predictions - self.y
        partial_derivative = difference.transpose() * self.X
        return partial_derivative.transpose()
        
def _map(Matrix, func):
    def map_single(element):
        return func(element)
                             
    vec = np.vectorize(map_single)
    return vec(Matrix)

def _sigmoid(x):
    return 1.0 / (1 + m.exp(-x))
