from data_processor import DataProcessor
from convex_function import LogisticRegression
from convex_optimization import GradientDescent
from copy import deepcopy
import numpy as np


class ModelTrainer:
    TRAINING_TEST_DATA_SPLIT_RATIO = 0.7
    OUTPUT_FILE = "trained_thetas.txt"

    def __init__(self):
        pass

    # each model function will pre-process data differently in order to
    # help us find the best model

    # model 1 : no pre processing
    def model_1_pre_process(self, X):
        X = deepcopy(X)
        X = DataProcessor.normalize_features(X, True)

        return X

    # model 2 : convert input to greyscale
    def model_2_pre_process(self, X):
        import pdb; pdb.set_trace()
        X = deepcopy(X)
        # remove intercept column
        X = np.delete(X, 0, 1)

        # convert RGB to greyscale
        X = X.tolist()
        newX = []
        for row in X:
            reds = row[0::3]
            greens = row[1::3]
            blues = row[2::3]
            rgb_row = zip(reds, greens, blues)
            # greyscale value = 0.21R + 0.72G + 0.07B
            rgb_row = map(lambda x: 0.21 * x[0] + 0.72 * x[1] + 0.07 * x[2],
                          rgb_row)
            newX.append(rgb_row)
        X = np.matrix(newX)
        newX = []
        X = np.insert(X, 0, 1, 1)
        X = DataProcessor.normalize_features(X, True)

        return X

    def train_models(self):
        MODELS_TO_TRAIN = [self.model_2_pre_process, self.model_1_pre_process]

        optimal_thetas = [[] for _ in xrange(len(MODELS_TO_TRAIN))]
        for i, model_preprocessor in enumerate(MODELS_TO_TRAIN):
            processor = DataProcessor()
            import pdb; pdb.set_trace()
            processor.load_input()
            processor.load_output()

            n = processor.input.shape[1]

            theta_init = np.zeros(n + 1)
            theta_init = np.matrix(theta_init)
            theta_init = theta_init.transpose()

            # add intercept term
            processor.input = np.insert(processor.input, 0, 1, 1)
            processor.input = processor.input.astype('float64')
            processor.input = model_preprocessor(processor.input)

            processor.split_to_training_test(
                self.TRAINING_TEST_DATA_SPLIT_RATIO)

            X = processor.training_input
            for j in xrange(processor.num_labels):

                y = (processor.training_output == j)
                y = y.astype(int)

                model = LogisticRegression(theta_init, X, y)
                optimizer = GradientDescent(model)

                theta_optimal = optimizer.find_min()
                theta_optimal = theta_optimal.transpose()

                optimal_thetas[i].append(theta_optimal.tolist()[0])

        # theta indices i,j specify ith model and jth classification type
        with open(self.OUTPUT_FILE, "w") as f:
            for i, thetas in enumerate(optimal_thetas):
                for j, theta in enumerate(thetas):
                    theta_str = ','.join(map(str, theta))
                    f.write("theta_%d_%d=%s\n" % (i, j, theta_str))

if __name__ == '__main__':
    ModelTrainer().train_models()
