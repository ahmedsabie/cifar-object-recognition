xrange = range
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from convex_function import LogisticRegression
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

class ModelEvaluator:
    FILE = ModelTrainer.OUTPUT_FILE
    LOGISTIC_MODELS = [ModelTrainer.model_1_pre_process,
              ModelTrainer.model_2_pre_process]

    def __init__(self):
        pass

    def evaluate_logistic_models(self):
        with open(self.FILE, "r") as f:
            for i, model_preprocessor in enumerate(self.LOGISTIC_MODELS):
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
                print (output)

    def evaluate_cnn_models(self):
        data = DataProcessor()
        data.load_input(is_cnn_model=True)
        data.load_output()
        data.output = data.output.transpose()
        data.output = np.array(data.output)[0]

        batch_size = 128
        epochs = 200

        # input image dimensions
        img_rows, img_cols = DataProcessor.IMAGE_SIZE, DataProcessor.IMAGE_SIZE

        input_shape = (img_rows, img_cols, 3)

        data.input = data.input.astype('float32')
        data.input = data.input / 255;
        #data.input = data.normalize_features(data.input, False)
        data.split_to_training_test(
            ModelTrainer.TRAINING_TEST_DATA_SPLIT_RATIO)
        x_train = data.training_input
        x_test = data.test_input

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(data.training_output,
                                             data.NUM_LABELS)
        y_test = keras.utils.to_categorical(data.test_output,
                                            data.NUM_LABELS)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3),
                         border_mode='same',
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(128, kernel_size=(3, 3),
                         border_mode='same',
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(data.NUM_LABELS, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

if __name__ == '__main__':
    #ModelTrainer().train_models()
    #ModelEvaluator().evaluate_logistic_models_models()
    ModelEvaluator().evaluate_cnn_models()
