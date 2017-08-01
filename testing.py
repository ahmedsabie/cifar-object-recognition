import unittest
import sys
from convex_function import LogisticRegression
import numpy as np

FILE = 'logistic_regression_tests_input.txt'
EPS = 0.1

class LogisticRegressionTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LogisticRegressionTests, self).__init__(*args, **kwargs)
        self.read_input()
        self.create_model()
        
    def read_input(self):
        self.inputs = {}
        
        with open(FILE, 'r') as inputs:
            for line in inputs:
                line = line.strip()
                if not line:
                    continue
                line = line.split('=')
                if line:
                    var = line[0]
                    val = eval(line[-1].strip())
                    self.inputs[var] = val
                    
        for var, key in self.inputs.iteritems():
            self.inputs[var] = np.matrix(key)

    def create_model(self):
        inputs = self.inputs
        self.model = LogisticRegression(inputs['theta_init'],
                                        inputs['X'],
                                        inputs['y'])
        
    def test_compute_predictions(self):
        comparator = abs(self.model.compute_predictions() - \
                         self.inputs['predictions']) < EPS
        self.assertTrue(comparator.all())

    def test_cost_function(self):
        comparator = abs(self.model.cost_function() - self.inputs['J']) < EPS
        self.assertTrue(comparator)

    def test_cost_function_gradient(self):
        comparator = abs(self.model.cost_function_gradient() - \
                         self.inputs['J_gradient']) < EPS
        self.assertTrue(comparator.all())
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
