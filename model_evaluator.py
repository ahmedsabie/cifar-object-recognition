from data_processor import DataProcessor
from model_trainer import ModelTrainer
from convex_function import LogisticRegression
import numpy as np


class ModelEvaluator:
    FILE = ModelTrainer.OUTPUT_FILE
    MODELS = [ModelTrainer.model_1_pre_process,
              ModelTrainer.model_2_pre_process]

    def __init__(self):
        pass

    def evaluate_models(self):
        with open(self.FILE, "r") as f:
            for i, model_preprocessor in enumerate(self.MODELS):
                processor = DataProcessor()
                processor.load_input()
                processor.load_output()

                processor.input = np.insert(processor.input, 0, 1, 1)
                processor.input = processor.input.astype('float64')
                processor.input = model_preprocessor(processor.input)

                m = processor.input.shape[0]
                processor.split_to_training_test(
                    ModelTrainer.TRAINING_TEST_DATA_SPLIT_RATIO)

                probabilities_training = []
                probabilities_test = []
                for j in xrange(processor.num_labels):
                    input = f.readline().strip().split('=')
                    input = input[-1].split(',')
                    theta = map(float, input)
                    theta = np.matrix(theta)
                    theta = theta.transpose()


                    func_training = LogisticRegression(theta,
                                                     processor.training_input,
                                                     processor.training_output)
                    predictions_training = func_training.compute_predictions()
                    probabilities_training.append(
                        predictions_training.transpose().tolist()[0])

                    func_test = LogisticRegression(theta,
                                                       processor.test_input,
                                                       processor.test_output)
                    predictions_test = func_test.compute_predictions()
                    probabilities_test.append(
                        predictions_test.transpose().tolist()[0])

                # get labels
                labels_training = np.argmax(probabilities_training, axis=0)
                labels_training = np.matrix(labels_training).transpose()

                labels_test = np.argmax(probabilities_test, axis=0)
                labels_test = np.matrix(labels_test).transpose()

                # calculate accuracy
                correct_training = \
                    (labels_training == processor.training_output).sum()

                correct_test = \
                    (labels_test == processor.test_output).sum()

                accuracy_training = correct_training * 100.0 / m
                accuracy_test = correct_test * 100.0 / m

                output = 'Model %d has %.2f%% accuracy on training set and ' \
                         '%.2f%% accuracy on test set'
                output = output % (i+1, accuracy_training, accuracy_test)
                print output

if __name__ == '__main__':
    #ModelTrainer().train_models()
    ModelEvaluator().evaluate_models()
